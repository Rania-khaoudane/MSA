# Numbers for the manuscript

All correctness values use the validated judge.

- Judge: Atlas-Chat-9B / v2; Cohen's kappa = 0.797 against human labels (raw agreement 0.906, parse rate 1.000).
- Dialect gap in answers (k=5): -0.120 [-0.190, -0.055], Holm-adjusted p = 0.002 (significant).
- Cost of k=1 without reranking: -0.130 [-0.200, -0.060], Holm-adjusted p = 0.003 (significant).
- Reranking at k=1: +0.145 [+0.095, +0.195], Holm-adjusted p = 0.002 (significant).
- Reranked k=1 vs retrieval k=5: +0.015 [-0.050, +0.080], Holm-adjusted p = 1.000 (not significant).
- Reranking at k=5: -0.005 [-0.060, +0.050], Holm-adjusted p = 1.000 (not significant).
- Normalisation, end to end: -0.055 [-0.115, +0.010], Holm-adjusted p = 0.452 (not significant).
- Gating vs always reranking (k=1): -0.010 [-0.025, +0.000], Holm-adjusted p = 0.852 (not significant).
- Headroom to the oracle: +0.060 [+0.025, +0.100], Holm-adjusted p = 0.010 (significant).
- Prompt length, retrieval k=5 vs reranked k=1: 3.7x shorter.
- Power: at n = 200 paired questions, the minimum detectable difference in correctness (alpha 0.05, power 0.80) is 0.096.
- Refusals: 275 of 1600 answers; faithfulness is reported on non-refused answers only, and refusal rate separately.
- Oracle: not significantly below any condition; it behaves as a ceiling.
- Second generator: 4/4 comparisons keep their direction and 3/4 their significance with Atlas-Chat-9B as generator.
- Prompt sensitivity, dialect gap: -0.135 to -0.115 across 3 prompts; significant in 3/3; direction consistent.
- Prompt sensitivity, reranked k=1 vs k=5: +0.015 to +0.070 across 3 prompts; significant in 1/3; direction consistent.
