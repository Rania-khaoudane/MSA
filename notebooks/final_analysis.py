# =============================================================================
# Final analysis: every reported number, from the validated judge only.
#
# Reads the re-judged results and writes one folder, final_results/, that holds
# everything the manuscript reports. Nothing here uses the old judge. Runs on
# CPU in seconds.
#
#   Inputs
#     judge_improvement/rag_generation_raw_rejudged.csv  main run A-H
#     judge_improvement/judge_selection.csv              kappa per judge config
#     rejudged/second_generator_rejudged.csv             point K
#     rejudged/prompt_sensitivity_rejudged.csv           point G
#     rag_outputs/rag_retrieval_summary.csv              retrieval (judge-free)
#     dev_test_outputs/test_split_retrieval.csv          point H (judge-free)
#
#   Outputs (final_results/)
#     tables/*.tex           LaTeX table rows, ready to \input
#     *.csv                  every underlying number
#     fig6_rag.pdf/.png      end-to-end figure
#     PAPER_NUMBERS.md       paste-ready sentences for the manuscript
#     checksums.json         SHA-256 of every file above
#
# RUN:  python3 final_analysis.py
# =============================================================================

CONFIG = {
    "main": "judge_improvement/rag_generation_raw_rejudged.csv",
    "judge_selection": "judge_improvement/judge_selection.csv",
    "second_generator": "rejudged/second_generator_rejudged.csv",
    "prompt_sensitivity": "rejudged/prompt_sensitivity_rejudged.csv",
    "retrieval": "rag_outputs/rag_retrieval_summary.csv",
    "dev_test": "dev_test_outputs/test_split_retrieval.csv",
    "qa_file": "qa_pairs_wiki.json",
    "out": "final_results",
    "bootstrap_n": 4000,
    "seed": 42,
}

import os, re, json, hashlib
import numpy as np
import pandas as pd

OUT = CONFIG["out"]
TAB = os.path.join(OUT, "tables")
os.makedirs(TAB, exist_ok=True)
rng = np.random.default_rng(CONFIG["seed"])
def have(p): return os.path.exists(p)
def o(n): return os.path.join(OUT, n)
notes = []                       # sentences for PAPER_NUMBERS.md
def note(s): notes.append(s); print(s)

LABEL = {"A": "MSA · retrieval · k=5", "B": "Darija · retrieval · k=5",
         "C": "Darija · retrieval · k=1", "D": "Darija · normalised · k=5",
         "E": "Darija · reranked · k=5", "F": "Darija · reranked · k=1",
         "G": "Darija · gated · k=1", "H": "Darija · gold passage"}

# ---------------------------------------------------------------- helpers
DIAC = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED\u0670]")
def tok(t):
    t = DIAC.sub("", str(t or ""))
    for a, b in [(r"[\u0625\u0623\u0622\u0627]", "\u0627"), (r"\u0649", "\u064A"),
                 (r"\u0629", "\u0647"), (r"\u0624", "\u0648"), (r"\u0626", "\u064A"),
                 (r"\u0640+", ""), (r"[^\w\s]", " ")]:
        t = re.sub(a, b, t)
    return re.sub(r"\s+", " ", t).strip().split()
def f1(pred, gold):
    p, g = tok(pred), tok(gold)
    common = sum(min(p.count(w), g.count(w)) for w in set(p)) if p and g else 0
    if not common: return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)

def boot(d):
    d = np.asarray(d, float)
    m = d[rng.integers(0, len(d), size=(CONFIG["bootstrap_n"], len(d)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    p = min(max(2 * min((m <= 0).mean(), (m >= 0).mean()), 1 / CONFIG["bootstrap_n"]), 1.0)
    return d.mean(), lo, hi, p

def paired(df, a, b, col="correct", key="qid"):
    x = df[df.cond == a].set_index(key)[col]; y = df[df.cond == b].set_index(key)[col]
    i = x.index.intersection(y.index)
    return boot((x.loc[i] - y.loc[i]).values)

def holm(p):
    p = np.asarray(p, float); idx = np.argsort(p); m = len(p); run = 0.0; adj = np.empty(m)
    for i, v in enumerate(p[idx]):
        run = max(run, (m - i) * v); adj[i] = min(run, 1.0)
    out = np.empty(m); out[idx] = adj; return out

def ci(d, lo, hi): return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
def tex(name, lines): open(os.path.join(TAB, name), "w").write("\n".join(lines) + "\n")

qa = {q["id"]: q for q in json.load(open(CONFIG["qa_file"], encoding="utf-8"))}

# ---------------------------------------------------------------- judge validity
print("=" * 78); print("JUDGE VALIDITY"); print("=" * 78)
if have(CONFIG["judge_selection"]):
    js = pd.read_csv(CONFIG["judge_selection"]).sort_values("kappa", ascending=False)
    js.to_csv(o("judge_selection.csv"), index=False)
    best = js.iloc[0]
    note(f"- Judge: {best.config}; Cohen's kappa = {best.kappa:.3f} against human labels "
         f"(raw agreement {best.agreement:.3f}, parse rate {best.parse_rate:.3f}).")
    tex("table_judge.tex", [f"{r.config} & {r.kappa:.3f} & {r.agreement:.3f} & {r.parse_rate:.3f}\\\\"
                            for _, r in js.iterrows()])

# ---------------------------------------------------------------- main results
print("\n" + "=" * 78); print("MAIN RESULTS (A-H)"); print("=" * 78)
main = pd.read_csv(CONFIG["main"])
main["correct"], main["faithful"] = main["correct_v2"], main["faithful_v2"]
if "refused" not in main:
    main["refused"] = main.answer.fillna("").str.contains("غير متوفرة").astype(int)
main["token_f1"] = [0.0 if r else f1(a, qa[q]["gold_answer"])
                    for a, q, r in zip(main.answer, main.qid, main.refused)]
main.to_csv(o("main_results_raw.csv"), index=False)

order = [c for c in LABEL if c in set(main.cond)]
S = main.groupby("cond").agg(n=("qid", "count"), gold_in_context=("gold_in_context", "mean"),
                             correct=("correct", "mean"), token_f1=("token_f1", "mean"),
                             refusal=("refused", "mean"), prompt_tokens=("prompt_tokens", "mean")
                             ).reindex(order)
S["faithful_answered"] = main[main.refused == 0].groupby("cond").faithful.mean()
S.insert(0, "condition", [LABEL[c] for c in S.index])
S.to_csv(o("main_summary.csv"))
print(S.to_string(float_format=lambda x: f"{x:.3f}"))
tex("table6_rag.tex", [f"{LABEL[c].replace('·', '&')} & {r.gold_in_context:.3f} & {r.correct:.3f} & "
                       f"{r.faithful_answered:.3f} & {r.token_f1:.3f} & {r.prompt_tokens:,.0f}\\\\".replace(",", "{,}")
                       for c, r in S.iterrows()])

# ---------------------------------------------------------------- comparisons + Holm
print("\n" + "=" * 78); print("PAIRED COMPARISONS, HOLM-CORRECTED"); print("=" * 78)
COMPARE = [("B", "A", "Dialect gap in answers (k=5)"), ("C", "B", "Cost of k=1 without reranking"),
           ("F", "C", "Reranking at k=1"), ("F", "B", "Reranked k=1 vs retrieval k=5"),
           ("E", "B", "Reranking at k=5"), ("D", "B", "Normalisation, end to end"),
           ("G", "F", "Gating vs always reranking (k=1)"), ("H", "F", "Headroom to the oracle")]
rows = []
for a, b, lab in COMPARE:
    if a in order and b in order:
        for col in ["correct", "token_f1"]:
            d, lo, hi, p = paired(main, a, b, col)
            rows.append({"comparison": lab, "a": a, "b": b, "metric": col,
                         "diff": d, "lo": lo, "hi": hi, "p_raw": p})
C = pd.DataFrame(rows)
C["p_holm"] = np.nan
for m, g in C.groupby("metric"):
    C.loc[g.index, "p_holm"] = holm(g.p_raw.values)
C["sig_holm"] = np.where(C.p_holm < 0.05, "yes", "no")
C.to_csv(o("comparisons.csv"), index=False)
print(C.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
cc = C[C.metric == "correct"]
tex("table_comparisons.tex", [f"{r.comparison} & {ci(r['diff'], r.lo, r.hi)} & {r.p_holm:.3f} & {r.sig_holm}\\\\"
                              for _, r in cc.iterrows()])
for _, r in cc.iterrows():
    note(f"- {r.comparison}: {ci(r['diff'], r.lo, r.hi)}, Holm-adjusted p = {r.p_holm:.3f} "
         f"({'significant' if r.sig_holm == 'yes' else 'not significant'}).")
if {"B", "F"} <= set(S.index):
    note(f"- Prompt length, retrieval k=5 vs reranked k=1: "
         f"{S.loc['B', 'prompt_tokens'] / S.loc['F', 'prompt_tokens']:.1f}x shorter.")

# ---------------------------------------------------------------- power
d = main[main.cond == "B"].set_index("qid").correct - main[main.cond == "A"].set_index("qid").correct
mde = (1.959964 + 0.841621) * d.dropna().std(ddof=1) / np.sqrt(d.notna().sum())
note(f"- Power: at n = {d.notna().sum()} paired questions, the minimum detectable difference in "
     f"correctness (alpha 0.05, power 0.80) is {mde:.3f}.")

# ---------------------------------------------------------------- faithfulness + refusal
ref = main[main.refused == 1]
note(f"- Refusals: {len(ref)} of {len(main)} answers; faithfulness is reported on non-refused "
     f"answers only, and refusal rate separately.")

# ---------------------------------------------------------------- oracle vs every condition
print("\n" + "=" * 78); print("ORACLE vs EVERY CONDITION"); print("=" * 78)
orows = []
for c in order:
    if c == "H": continue
    d_, lo, hi, p = paired(main, "H", c)
    orows.append({"vs": c, "diff": d_, "lo": lo, "hi": hi, "p": p,
                  "verdict": "oracle worse" if hi < 0 else ("oracle better" if lo > 0 else "n.s.")})
Ob = pd.DataFrame(orows); Ob.to_csv(o("oracle_check.csv"), index=False)
print(Ob.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
worse = Ob[Ob.verdict == "oracle worse"]
note("- Oracle: " + ("scores significantly below " + ", ".join(worse.vs) + " -- needs discussion."
                     if len(worse) else "not significantly below any condition; it behaves as a ceiling."))

# ---------------------------------------------------------------- second generator
print("\n" + "=" * 78); print("SECOND GENERATOR (point K)"); print("=" * 78)
if have(CONFIG["second_generator"]):
    g2 = pd.read_csv(CONFIG["second_generator"])
    grows = []
    for a, b, lab in [("B", "A", "Dialect gap in answers"), ("F", "C", "Reranking at k=1"),
                      ("C", "B", "Cost of k=1 without reranking"), ("H", "F", "Headroom to the oracle")]:
        if {a, b} <= set(g2.cond) and {a, b} <= set(main.cond):
            d1, l1, h1, _ = paired(main, a, b); d2, l2, h2, _ = paired(g2, a, b)
            s1, s2 = (l1 > 0 or h1 < 0), (l2 > 0 or h2 < 0)
            grows.append({"comparison": lab, "qwen": ci(d1, l1, h1), "atlas_chat": ci(d2, l2, h2),
                          "same_direction": (d1 > 0) == (d2 > 0), "same_significance": s1 == s2})
    G = pd.DataFrame(grows); G.to_csv(o("second_generator_agreement.csv"), index=False)
    print(G.to_string(index=False))
    tex("table_second_generator.tex", [f"{r.comparison} & {r.qwen} & {r.atlas_chat}\\\\" for _, r in G.iterrows()])
    note(f"- Second generator: {int(G.same_direction.sum())}/{len(G)} comparisons keep their direction and "
         f"{int(G.same_significance.sum())}/{len(G)} their significance with Atlas-Chat-9B as generator.")
else:
    print("  not found -- run rejudge_all.py")

# ---------------------------------------------------------------- prompt sensitivity
print("\n" + "=" * 78); print("PROMPT SENSITIVITY (point G)"); print("=" * 78)
if have(CONFIG["prompt_sensitivity"]):
    ps = pd.read_csv(CONFIG["prompt_sensitivity"])
    piv = ps.pivot_table(index="cond", columns="prompt", values="correct", aggfunc="mean")
    piv["spread"] = piv.max(axis=1) - piv.min(axis=1)
    piv.to_csv(o("prompt_sensitivity.csv"))
    print(piv.to_string(float_format=lambda x: f"{x:.3f}"))
    prows = []
    for pid in sorted(ps.prompt.unique()):
        sub = ps[ps.prompt == pid]
        for a, b, lab in [("B", "A", "dialect gap"), ("F", "B", "reranked k=1 vs k=5")]:
            if {a, b} <= set(sub.cond):
                d_, lo, hi, _ = paired(sub, a, b)
                prows.append({"comparison": lab, "prompt": pid, "diff": d_, "lo": lo, "hi": hi,
                              "significant": lo > 0 or hi < 0})
    P = pd.DataFrame(prows); P.to_csv(o("prompt_sensitivity_comparisons.csv"), index=False)
    print(P.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    tex("table7_prompt_sensitivity.tex",
        [f"{c} & " + " & ".join(f"{piv.loc[c, p]:.3f}" for p in piv.columns if p != "spread")
         + f" & {piv.loc[c, 'spread']:.3f}\\\\" for c in piv.index])
    for lab, g in P.groupby("comparison"):
        note(f"- Prompt sensitivity, {lab}: {g['diff'].min():+.3f} to {g['diff'].max():+.3f} across "
             f"{len(g)} prompts; significant in {int(g.significant.sum())}/{len(g)}; direction "
             f"{'consistent' if (g['diff'] > 0).all() or (g['diff'] < 0).all() else 'NOT consistent'}.")
else:
    print("  not found -- run rejudge_all.py")

# ---------------------------------------------------------------- judge-free results
for key, fname in [("retrieval", "retrieval_summary.csv"), ("dev_test", "dev_test_split.csv")]:
    if have(CONFIG[key]):
        pd.read_csv(CONFIG[key]).to_csv(o(fname), index=False)

# ---------------------------------------------------------------- figure 6
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
                     "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.titlepad": 17, "savefig.bbox": "tight", "savefig.dpi": 300})
COL = {"A": "#0072B2", "B": "#D55E00", "C": "#D55E00", "D": "#CC79A7",
       "E": "#009E73", "F": "#009E73", "G": "#E69F00", "H": "#8a8a8a"}
K = {"A": 5, "B": 5, "C": 1, "D": 5, "E": 5, "F": 1, "G": 1, "H": 1}
fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(order) + 1.4))
for i, c in enumerate(order):
    m, lo, hi, _ = boot(main[main.cond == c].correct.values)
    ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=COL[c], ms=8, capsize=3,
                mfc=COL[c] if K[c] == 1 else "white", mew=2)
    ax.text(hi + 0.012, i, f"{m:.3f}  ·  {S.loc[c, 'prompt_tokens']:,.0f} tok", va="center",
            fontsize=8, color="#444")
ax.set_yticks(range(len(order))); ax.set_yticklabels([LABEL[c] for c in order]); ax.invert_yaxis()
ax.set_xlabel("Answer correctness (validated judge, 95% CI)")
ax.set_xlim(max(0, main.correct.mean() - 0.4), 1.1)
ax.set_title("End-to-end answer quality")
ax.text(0, 1.012, "Filled marker = one passage in context; label shows mean prompt length",
        transform=ax.transAxes, fontsize=8.3, color="#8a8a8a", va="bottom")
for ext in ("pdf", "png"):
    fig.savefig(o(f"fig6_rag.{ext}"))
plt.close(fig)

# ---------------------------------------------------------------- write-up + checksums
with open(o("PAPER_NUMBERS.md"), "w", encoding="utf-8") as f:
    f.write("# Numbers for the manuscript\n\nAll correctness values use the validated judge.\n\n")
    f.write("\n".join(notes) + "\n")

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()
files = sorted(os.path.join(r, n) for r, _, fs in os.walk(OUT) for n in fs if n != "checksums.json")
json.dump({f: sha(f) for f in files}, open(o("checksums.json"), "w"), indent=1)

print("\n" + "=" * 78)
print(f"Wrote {OUT}/ -- start with {OUT}/PAPER_NUMBERS.md")
print("=" * 78)
