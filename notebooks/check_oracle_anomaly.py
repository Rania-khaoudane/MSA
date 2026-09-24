# =============================================================================
# Review follow-up: is the oracle scoring below other conditions real or noise?
#
# The canonical run showed H (oracle, correct=0.750) below B (0.780) and
# E (0.770), despite the oracle guaranteeing gold-passage presence. This tests
# every condition against H directly, not just the one comparison the main
# script happened to include, so the anomaly is either confirmed or ruled out
# rather than left implicit.
#
# RUN after rag_end_to_end.py. No GPU needed -- this only reads its output CSV.
# =============================================================================
import numpy as np
import pandas as pd

CONFIG = {
    "gen_raw": "rag_outputs/rag_generation_raw.csv",
    "bootstrap_n": 4000,
    "seed": 42,
}
rng = np.random.default_rng(CONFIG["seed"])

df = pd.read_csv(CONFIG["gen_raw"])
if "H" not in set(df.cond):
    raise SystemExit("No oracle condition ('H') found in this file.")

def paired(a, b, col="correct"):
    x = df[df.cond == a].set_index("qid")[col]
    y = df[df.cond == b].set_index("qid")[col]
    i = x.index.intersection(y.index)
    d = (x.loc[i] - y.loc[i]).values.astype(float)
    m = d[rng.integers(0, len(d), size=(CONFIG["bootstrap_n"], len(d)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return d.mean(), lo, hi, len(i)

print("=" * 78)
print("ORACLE (H) vs EVERY OTHER CONDITION")
print("=" * 78)
print(f"{'condition':<12}{'metric':<10}{'H - other':>12}  {'95% CI':<20}{'n':>6}  sig")
rows = []
for other in sorted(set(df.cond) - {"H"}):
    for metric in ["correct", "token_f1"]:
        d, lo, hi, n = paired("H", other, metric)
        sig = "H WORSE" if hi < 0 else ("H better" if lo > 0 else "n.s.")
        print(f"{other:<12}{metric:<10}{d:>+12.3f}  [{lo:+.3f}, {hi:+.3f}]{'':<3}{n:>6}  {sig}")
        rows.append({"other": other, "metric": metric, "diff": d, "lo": lo, "hi": hi, "n": n, "verdict": sig})

R = pd.DataFrame(rows)
R.to_csv("oracle_check.csv", index=False)

worse = R[(R.metric == "correct") & (R.verdict == "H WORSE")]
print("\n" + "=" * 78)
if len(worse):
    print(f"CONFIRMED: oracle scores significantly below {list(worse.other)} on correctness.")
    print("This needs a real explanation in Discussion, not a one-line caveat.")
    print("Likely causes to check by hand:")
    print("  - re-read 10-15 oracle transcripts where correct=0: is the judge")
    print("    penalising a right answer for wording, or is the answer genuinely wrong")
    print("    despite having the correct passage?")
    print("  - the oracle context is a single, very short passage: check whether the")
    print("    generator is truncating or hedging more when given less context to work with")
else:
    print("NOT CONFIRMED: no condition scores significantly above the oracle on")
    print("correctness. The point estimate below H is consistent with sampling noise")
    print("at n=200; one sentence in Limitations is sufficient.")

print(f"\nsaved oracle_check.csv ({len(R)} rows)")

# A quick look at where the oracle actually loses, for the manual read-through above.
lost = df[(df.cond == "H") & (df.correct == 0)]
print(f"\n{len(lost)} oracle items scored incorrect. First 10 qids for manual review:")
print(list(lost.qid.head(10)))
