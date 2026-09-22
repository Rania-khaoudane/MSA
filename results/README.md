# Results

CSV outputs from the notebook runs. Upload the files you downloaded from Colab
here so the reported numbers can be traced to the run that produced them.

| File | Produced by | Serves |
|---|---|---|
| `results_pilot.csv`, `results_large.csv` | 03 retrieval baseline | The core dialect gap: MSA vs. Darija Recall@1/5/10 and MRR on the pilot and full corpus. |
| `encoder_grid.csv`, `recovery_no_api.csv` | 04 encoders + rule-based | Whether rule-based dialect normalisation recovers any of the retrieval loss, across non-Arabic-specific encoders. |
| `results_with_ci.csv`, `dialect_gaps.csv`, `mitigation_ci.csv` | 05 confound + CIs | Confirms the gap and mitigation results hold up statistically (bootstrap CIs) and are not an artefact of the pilot corpus's authorship confound. |
| `arabic_encoder_sweep.csv`, `arabic_gaps.csv`, `arabic_mitigation.csv` | 06 Arabic encoders | Whether Arabic-specific embedding models close the gap better than general multilingual ones. |
| `finetune_results.csv`, `finetune_by_subset.csv` | 07 fine-tuning | Whether contrastive fine-tuning on parallel MSA/Darija pairs improves retrieval. |
| `robustness_raw.csv`, `robustness_per_subset.csv`, `robustness_ablation.csv` | 08 robustness | Whether the fine-tuning result is stable across seeds and subsets, or an overfitting artefact. |
| `wiki_only_raw.csv` | 09 Wikipedia-only | Whether fine-tuning still helps when trained only on the (confound-free) Wikipedia items. |
| `generation_*_raw.csv`, `generation_*_summary.csv`, `generation_*_comparisons.csv` | 10 generation | Whether the retrieval-stage dialect gap actually reaches the final generated answer, end-to-end. |
| `reranker_experiments_full.csv`, `gap_by_reranker.csv`, `asymmetry_test.csv` | 11 reranker experiments | Confirms cross-encoder reranking is a real fix (not a scoring bug), compares general vs. Arabic-specific rerankers, and tests whether the fix helps dialect queries more than MSA. |
| `k_sweep_summary.csv`, `k_sweep_comparisons.csv`, `k_sweep_raw.csv` | 12 k-sweep | Whether shrinking the number of passages fed to the generator (fewer distractors) explains the oracle-vs-reranked correctness gap. |
| `latency_measurements.csv` | 13 latency benchmark | The real per-query cost of adding a reranking stage, used to justify why reranking every query isn't free. |
| `margin_gating_curve.csv`, `margin_gating_operating_points.csv`, `margin_gating_gap.csv` | 14 margin-gated reranking | The proposed cost-saving fix: reranking only low-confidence queries, and how much of the accuracy benefit is kept at a fraction of the compute. |

*Note: an earlier reranking run (`rerank_results.csv`, `rerank_gains.csv`,
`rerank_dialect_gap.csv`) tested a different model line-up and is superseded
by notebook 11 above.*
