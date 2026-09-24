# =============================================================================
# Judge-free check of the second-generator result.
#
# The Atlas-Chat-9B answers were scored by Atlas-Chat-9B itself (self-preference
# risk), and the only other judge available, Qwen2.5-14B, is weak (kappa 0.54).
# Token-F1 against the gold answer needs no judge at all, so it settles whether
# the second-generator conclusions hold independently of who does the judging.
#
# CPU only, a few seconds.  RUN:  python3 second_generator_check.py
# =============================================================================
import json, re
import numpy as np
import pandas as pd

MAIN = "final_results/main_results_raw.csv"          # Qwen answers, token_f1 already computed
SECOND = "rejudged/second_generator_rejudged.csv"    # Atlas-Chat answers
QA = "qa_pairs_wiki.json"
OUT = "final_results/second_generator_tokenf1.csv"
rng = np.random.default_rng(42)

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
    if not common:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)

qa = {q["id"]: q for q in json.load(open(QA, encoding="utf-8"))}
main = pd.read_csv(MAIN)
sec = pd.read_csv(SECOND)
if "refused" not in sec:
    sec["refused"] = sec.answer.fillna("").str.contains("غير متوفرة").astype(int)
sec["token_f1"] = [0.0 if r else f1(a, qa[q]["gold_answer"])
                   for a, q, r in zip(sec.answer, sec.qid, sec.refused)]

def paired(df, a, b, col):
    x = df[df.cond == a].set_index("qid")[col]
    y = df[df.cond == b].set_index("qid")[col]
    i = x.index.intersection(y.index)
    d = (x.loc[i] - y.loc[i]).values.astype(float)
    m = d[rng.integers(0, len(d), size=(4000, len(d)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return d.mean(), lo, hi

print("Mean token-F1 by condition (no judge involved)")
print(pd.DataFrame({"Qwen2.5-7B": main.groupby("cond").token_f1.mean(),
                    "Atlas-Chat-9B": sec.groupby("cond").token_f1.mean()}).dropna()
      .to_string(float_format=lambda x: f"{x:.3f}"))

rows = []
for a, b, lab in [("B", "A", "Dialect gap in answers"), ("F", "C", "Reranking at k=1"),
                  ("C", "B", "Cost of k=1 without reranking"), ("H", "F", "Headroom to the oracle")]:
    r = {"comparison": lab}
    for name, df in [("qwen", main), ("atlas", sec)]:
        if {a, b} <= set(df.cond):
            d, lo, hi = paired(df, a, b, "token_f1")
            r[f"{name}_tokenf1"] = f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            r[f"{name}_sig"] = bool(lo > 0 or hi < 0)
    d, lo, hi = paired(sec, a, b, "correct")
    r["atlas_selfjudged"] = f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    r["atlas_selfjudged_sig"] = bool(lo > 0 or hi < 0)
    rows.append(r)

R = pd.DataFrame(rows)
R.to_csv(OUT, index=False)
print("\nPaired comparisons: token-F1 (no judge) vs Atlas-Chat's self-judged correctness")
print(R.to_string(index=False))

print("\nHow to read this")
gap = R[R.comparison == "Dialect gap in answers"].iloc[0]
if gap.atlas_sig == gap.atlas_selfjudged_sig:
    print(f"  For Atlas-Chat, the dialect gap is "
          f"{'significant' if gap.atlas_sig else 'NOT significant'} both with and without a judge.")
    print("  Self-preference does not explain the result. Report token-F1 as the primary evidence.")
else:
    print("  For Atlas-Chat, the dialect-gap verdict CHANGES without a judge. The self-judged")
    print("  result is unreliable; report the token-F1 result and state the disagreement.")
agree = int((R.atlas_sig == R.atlas_selfjudged_sig).sum())
print(f"  Self-judged correctness and token-F1 agree on significance in {agree}/{len(R)} comparisons.")
print(f"\nWrote {OUT}")
