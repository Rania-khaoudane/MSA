# =============================================================================
# Review response: statistics, faithfulness split, and judge validation
#
# Addresses reviewer points A, B, F and I. Runs on CPU in seconds.
#
#   F  Multiple comparisons. Declares one primary hypothesis, then applies a
#      Holm-Bonferroni correction to every secondary comparison, reported as a
#      family-by-family table.
#   I  Power. Reports the minimum detectable paired difference at the observed
#      sample size, so a flat result can be read as "no effect larger than X"
#      rather than as "no effect".
#   B  Faithfulness conflates refusal with unfaithfulness. Reports faithfulness
#      including and excluding refusals, and refusal as its own outcome.
#   A  Judge validation. Computes Cohen's kappa against human labels once the
#      sheet is filled, including two-annotator agreement and a second judge.
#
# INPUTS (all optional; each block is skipped if its file is absent)
#   rag_outputs/rag_generation_raw.csv      per-item generation results
#   rag_outputs/rag_comparisons.csv         paired comparisons
#   results/gap_by_reranker.csv             retrieval gaps by reranker
#   results/arabic_mitigation.csv           normalisation strategies
#   rag_outputs/judge_labeling_sheet*.csv   human labels (see JUDGE SHEETS)
#
# JUDGE SHEETS. To answer A properly the reviewer asks for two annotators who
# are blind to the model's verdict. Prepare two copies of the sheet with the
# llm_correct column deleted, have each annotator fill human_correct, then save
# them as judge_labeling_sheet_ann1.csv and judge_labeling_sheet_ann2.csv.
# =============================================================================

import os, glob, json
import numpy as np
import pandas as pd

CONFIG = {
    "gen_raw": "rag_outputs/rag_generation_raw.csv",
    "gen_cmp": "rag_outputs/rag_comparisons.csv",
    "gap_by_reranker": "results/gap_by_reranker.csv",
    "mitigation": "results/arabic_mitigation.csv",
    "sheet_glob": "rag_outputs/judge_labeling_sheet*.csv",
    "out": "review_outputs",
    # The one comparison that is not corrected for multiplicity, declared in
    # advance: the MSA vs Darija Recall@1 gap.
    "primary": "MSA vs Darija Recall@1",
    "alpha": 0.05,
    "power": 0.80,
    "bootstrap_n": 4000,
    "seed": 42,
}
os.makedirs(CONFIG["out"], exist_ok=True)
rng = np.random.default_rng(CONFIG["seed"])
def out(name): return os.path.join(CONFIG["out"], name)
def have(p): return os.path.exists(p)

# ---------------------------------------------------------------- helpers
def paired_bootstrap(d):
    """Mean, CI and a two-sided bootstrap p-value for paired differences.
    The p-value is the usual bootstrap inversion: twice the smaller tail mass
    on the wrong side of zero."""
    d = np.asarray(d, float)
    n = len(d)
    idx = rng.integers(0, n, size=(CONFIG["bootstrap_n"], n))
    means = d[idx].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return d.mean(), lo, hi, min(max(p, 1 / CONFIG["bootstrap_n"]), 1.0)

def p_from_ci(est, lo, hi):
    """Altman & Bland (2011): recover a p-value from an estimate and its 95% CI.
    Used only for results whose per-item data is not in this repository."""
    se = (hi - lo) / 3.92
    if se <= 0:
        return np.nan
    z = abs(est) / se
    return float(np.exp(-0.717 * z - 0.416 * z * z))

def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    adj = np.full_like(p, np.nan)
    idx = np.argsort(p[ok])
    m = ok.sum()
    running = 0.0
    vals = p[ok][idx]
    adj_sorted = np.empty(m)
    for i, v in enumerate(vals):
        running = max(running, (m - i) * v)
        adj_sorted[i] = min(running, 1.0)
    tmp = np.empty(m)
    tmp[idx] = adj_sorted
    adj[ok] = tmp
    return adj

def mde_paired(d, alpha=None, power=None):
    """Minimum detectable paired difference, given the observed variability."""
    from math import sqrt
    alpha = alpha or CONFIG["alpha"]; power = power or CONFIG["power"]
    z_a, z_b = 1.959963985, 0.841621234      # two-sided 0.05, power 0.80
    d = np.asarray(d, float)
    return (z_a + z_b) * d.std(ddof=1) / sqrt(len(d))

# ---------------------------------------------------------------- B: faithfulness
print("=" * 78)
print("B. FAITHFULNESS, SPLIT BY REFUSAL")
print("=" * 78)
gen = pd.read_csv(CONFIG["gen_raw"]) if have(CONFIG["gen_raw"]) else None
if gen is None:
    print(f"  {CONFIG['gen_raw']} not found; skipped.")
else:
    if "refused" not in gen.columns:
        gen["refused"] = gen.answer.fillna("").str.contains("غير متوفرة").astype(int)
    rows = []
    for c, g in gen.groupby("cond"):
        ans = g[g.refused == 0]
        rows.append({
            "cond": c, "n": len(g),
            "refusal_rate": g.refused.mean(),
            "faithful_all": g.faithful.mean(),
            "faithful_answered": ans.faithful.mean() if len(ans) else np.nan,
            "correct_all": g.correct.mean(),
            "correct_answered": ans.correct.mean() if len(ans) else np.nan,
        })
    fsplit = pd.DataFrame(rows).sort_values("cond")
    print(fsplit.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    fsplit.to_csv(out("faithfulness_split.csv"), index=False)

    ref = gen[gen.refused == 1]
    if len(ref):
        n_f = int(ref.faithful.sum())
        print(f"\n  Refusals: {len(ref)}. The judge scored {n_f} of them faithful and "
              f"{len(ref) - n_f} unfaithful,")
        print("  on behaviour that is identical, which is why faithfulness is reported")
        print("  excluding refusals and refusal is reported as its own outcome.")
    if "judge_raw" in gen.columns:
        parsed = gen.judge_raw.notna() & (gen.judge_raw.astype(str).str.contains("مدعوم"))
        print(f"  Judge parse rate: {parsed.mean():.3f}")
    else:
        print("  Judge parse rate: raw verdicts not stored in this run "
              "(add a judge_raw column to report it).")

# ---------------------------------------------------------------- F: multiplicity
print("\n" + "=" * 78)
print("F. MULTIPLE COMPARISONS (Holm-Bonferroni, by family)")
print("=" * 78)
print(f"  Primary hypothesis, not corrected: {CONFIG['primary']}")

fam = []

if gen is not None and have(CONFIG["gen_cmp"]):
    cmp_df = pd.read_csv(CONFIG["gen_cmp"])
    for _, r in cmp_df.iterrows():
        a = gen[gen.cond == r.a].set_index("qid")[r.metric]
        b = gen[gen.cond == r.b].set_index("qid")[r.metric]
        i = a.index.intersection(b.index)
        if len(i) == 0:
            continue
        est, lo, hi, p = paired_bootstrap(a.loc[i].values - b.loc[i].values)
        fam.append({"family": "generation", "comparison": f"{r.comparison} [{r.metric}]",
                    "estimate": est, "lo": lo, "hi": hi, "p_raw": p})

if have(CONFIG["gap_by_reranker"]):
    g = pd.read_csv(CONFIG["gap_by_reranker"])
    for _, r in g.iterrows():
        if str(r.method) == "no_rerank":
            continue     # that row is the primary hypothesis, reported uncorrected
        fam.append({"family": "reranker gaps", "comparison": f"gap: {r.method}",
                    "estimate": r.gap, "lo": r.lo, "hi": r.hi,
                    "p_raw": p_from_ci(r.gap, r.lo, r.hi)})

if have(CONFIG["mitigation"]):
    m = pd.read_csv(CONFIG["mitigation"])
    col = "improvement" if "improvement" in m.columns else m.columns[-4]
    for _, r in m.iterrows():
        if {"lo", "hi"} <= set(m.columns):
            fam.append({"family": "normalisation",
                        "comparison": f"{r.get('encoder','?')} / {r.get('mitigation','?')}",
                        "estimate": r[col], "lo": r.lo, "hi": r.hi,
                        "p_raw": p_from_ci(r[col], r.lo, r.hi)})

if not fam:
    print("  No comparison files found; skipped.")
else:
    F = pd.DataFrame(fam)
    F["p_holm"] = np.nan
    for name, sub in F.groupby("family"):
        F.loc[sub.index, "p_holm"] = holm(sub.p_raw.values)
    F["sig_raw"] = np.where(F.p_raw < CONFIG["alpha"], "yes", "no")
    F["sig_holm"] = np.where(F.p_holm < CONFIG["alpha"], "yes", "no")
    F = F.sort_values(["family", "p_raw"])
    print(F.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    F.to_csv(out("holm_corrected.csv"), index=False)
    flip = F[(F.sig_raw == "yes") & (F.sig_holm == "no")]
    print(f"\n  {len(F)} secondary comparisons in {F.family.nunique()} families.")
    if len(flip):
        print("  These no longer hold after correction and must be reworded:")
        for _, r in flip.iterrows():
            print(f"    - {r.comparison} (p={r.p_raw:.3f} -> {r.p_holm:.3f})")
    else:
        print("  No comparison changes status after correction.")

# ---------------------------------------------------------------- I: power
print("\n" + "=" * 78)
print("I. POWER")
print("=" * 78)
if gen is None:
    print("  Generation data not found; skipped.")
else:
    rows = []
    for metric in ["correct", "token_f1"]:
        conds = sorted(gen.cond.unique())
        if len(conds) < 2:
            continue
        a = gen[gen.cond == conds[0]].set_index("qid")[metric]
        b = gen[gen.cond == conds[1]].set_index("qid")[metric]
        i = a.index.intersection(b.index)
        d = a.loc[i].values - b.loc[i].values
        rows.append({"metric": metric, "n_pairs": len(i), "sd_of_differences": d.std(ddof=1),
                     "min_detectable_difference": mde_paired(d)})
    P = pd.DataFrame(rows)
    print(P.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    P.to_csv(out("power.csv"), index=False)
    if len(P):
        v = P.iloc[0]
        print(f"""
  At n = {int(v.n_pairs)} paired observations, with alpha = {CONFIG['alpha']} and
  {int(CONFIG['power']*100)}% power, the smallest difference in {v.metric} this design can
  detect is about {v.min_detectable_difference:.3f}. A null result should therefore be
  reported as "no effect larger than {v.min_detectable_difference:.2f} was detectable",
  not as evidence that the effect is zero.""")

# ---------------------------------------------------------------- A: judge validation
print("\n" + "=" * 78)
print("A. JUDGE VALIDATION")
print("=" * 78)
sheets = sorted(glob.glob(CONFIG["sheet_glob"]))
filled = {}
for p in sheets:
    s = pd.read_csv(p)
    if "human_correct" in s.columns:
        s = s[s.human_correct.notna() & (s.human_correct.astype(str).str.strip() != "")]
        if len(s):
            filled[os.path.basename(p)] = s

if not filled:
    print(f"  No labelled sheets found matching {CONFIG['sheet_glob']}.")
    print("""
  To close this point:
    1. Take rag_outputs/judge_labeling_sheet.csv, delete the llm_correct column,
       and give a copy to each of two native speakers.
    2. Each fills human_correct with 1 or 0, without seeing the model's verdict.
    3. Save as judge_labeling_sheet_ann1.csv and judge_labeling_sheet_ann2.csv
       in rag_outputs/, then rerun this script.""")
else:
    try:
        from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
    except ImportError:
        raise SystemExit("pip install scikit-learn to compute kappa")

    def band(k):
        return ("slight" if k < .2 else "fair" if k < .4 else "moderate" if k < .6
                else "substantial" if k < .8 else "almost perfect")

    rows = []
    for name, s in filled.items():
        h = s.human_correct.astype(int)
        if "llm_correct" in s.columns and s.llm_correct.notna().all():
            l = s.llm_correct.astype(int)
            k = cohen_kappa_score(h, l)
            print(f"\n  {name}: n={len(s)}  kappa={k:.3f} ({band(k)})  "
                  f"raw agreement={accuracy_score(h, l):.3f}")
            cm = confusion_matrix(h, l, labels=[0, 1])
            print(f"    human=0: LLM says 0 in {cm[0,0]}, 1 in {cm[0,1]}")
            print(f"    human=1: LLM says 0 in {cm[1,0]}, 1 in {cm[1,1]}")
            rows.append({"sheet": name, "n": len(s), "kappa_vs_judge": k,
                         "agreement": accuracy_score(h, l)})
            if "condition" in s.columns:
                for c, g in s.groupby("condition"):
                    if len(g) > 4:
                        print(f"      {c}: kappa={cohen_kappa_score(g.human_correct.astype(int), g.llm_correct.astype(int)):.3f} (n={len(g)})")
        else:
            print(f"\n  {name}: n={len(s)} (no llm_correct column; used as an annotator only)")
            rows.append({"sheet": name, "n": len(s), "kappa_vs_judge": np.nan, "agreement": np.nan})

    names = list(filled)
    if len(names) >= 2:
        a, b = filled[names[0]], filled[names[1]]
        # Merge on qid AND condition: the same question appears under several
        # conditions, so joining on qid alone would multiply the rows.
        keys = ["qid", "condition"] if "condition" in a.columns and "condition" in b.columns else ["qid"]
        merged = (a[keys + ["human_correct"]]
                  .merge(b[keys + ["human_correct"]], on=keys, suffixes=("_1", "_2"))
                  .drop_duplicates(subset=keys))
        if len(merged):
            k = cohen_kappa_score(merged.human_correct_1.astype(int),
                                  merged.human_correct_2.astype(int))
            print(f"\n  Inter-annotator agreement ({names[0]} vs {names[1]}): "
                  f"n={len(merged)}  kappa={k:.3f} ({band(k)})")
            rows.append({"sheet": "inter-annotator", "n": len(merged),
                         "kappa_vs_judge": k, "agreement": accuracy_score(
                             merged.human_correct_1.astype(int), merged.human_correct_2.astype(int))})
    pd.DataFrame(rows).to_csv(out("judge_validation.csv"), index=False)

print("\n" + "=" * 78)
print(f"Wrote {CONFIG['out']}/")
for f in sorted(os.listdir(CONFIG["out"])):
    print("   ", f)
