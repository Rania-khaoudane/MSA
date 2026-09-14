# Dialect Mismatch in Arabic Retrieval-Augmented Generation

A benchmark, corpus and set of experiments measuring how Moroccan Darija queries
perform against a Modern Standard Arabic (MSA) knowledge base in a RAG pipeline —
at the retrieval stage and end-to-end.

**Findings summary:** [`docs/FINDINGS.md`](docs/FINDINGS.md)
**Full experimental record:** [`docs/EXPERIMENTAL_REPORT.md`](docs/EXPERIMENTAL_REPORT.md)

---

## Headline results

| | Result |
|---|---|
| Dialectal queries reduce **Recall@1** by 14.7 points (0.706 → 0.559) | 95% CI [0.090, 0.185] |
| Query normalisation (LLM, rule-based, expansion) recovers **none** of it | across 4 encoders × 4 strategies |
| Cause: normalisation collapses the gold-vs-competitor ranking margin | +0.040 → +0.009 |
| Contrastive fine-tuning closes **23%** of the Recall@1 gap | +0.070, 5/5 seeds, no MSA regression |
| The gap does **not** significantly reach generated answers at top-5 | +0.044, CI [−0.029, +0.118] |
| A single Arabic-only prompt constraint outperformed every retrieval-side fix | 0.824 → 0.882 correctness |

---

## Repository layout

```
data/
  corpus.json              3,054 passages (2,974 Wikipedia + 80 pilot)
  qa_pairs_wiki.json       200 Wikipedia-grounded MSA/Darija QA items
  qa_pairs_all.json        all 500 items (pilot + Wikipedia)
  pilot/
    corpus_pilot.json      the 80 author-written passages
    qa_pairs_pilot.json    the 300 pilot QA items
notebooks/                 the pipeline, in execution order
results/                   CSV outputs from the runs
docs/                      findings summary and full experimental report
LICENSE.md                 dual licensing — read before redistributing
```

### Data schema

Each QA item:

```json
{
  "id": "w043",
  "msa_query": "كم بلغ عدد سكان جماعة خنيفرة سنة 2024؟",
  "darija_query": "شحال وصل عدد سكان جماعة خنيفرة فـ2024؟",
  "gold_answer": "ارتفع عدد السكان إلى 123,738 نسمة حسب إحصاء 2024.",
  "source_chunk_id": "wiki_02194"
}
```

Each corpus passage carries a `source` field (`wikipedia_ar` or
`pilot_synthetic`) and, for Wikipedia passages, an `article_title` for
attribution.

---

## Notebooks

Run in order. All are Colab-ready; those marked GPU need a T4 or better.

| # | Notebook | Purpose | GPU |
|---|---|---|---|
| 01 | `01_corpus_builder.ipynb` | Build the Wikipedia corpus | no |
| 02 | `02_dataset_validation.ipynb` | Six-check benchmark validation | no |
| 03 | `03_retrieval_baseline.ipynb` | Three-condition retrieval baseline | no |
| 04 | `04_encoders_and_rulebased.ipynb` | Encoder sweep + rule-based mitigation | yes |
| 05 | `05_confound_and_confidence_intervals.ipynb` | Subset analysis, bootstrap CIs | yes |
| 06 | `06_arabic_encoders.ipynb` | GATE, AraBERT, Matryoshka encoders | yes |
| 07 | `07_finetune.ipynb` | Contrastive fine-tuning | yes |
| 08 | `08_finetune_robustness.ipynb` | Five seeds, ablation, per-subset | yes |
| 09 | `09_finetune_wikipedia_only.ipynb` | Clean test, no author-written data | yes |
| 10 | `10_generation_end_to_end.ipynb` | Generation + faithfulness, local LLM | yes |

`optional_qa_generator_fewshot.ipynb` generates additional QA items using the
hand-written set as few-shot examples.

**Reproducing the fine-tuned encoder.** The weights are not committed (size).
Notebook 07 reproduces them in roughly two minutes on a T4 from
`intfloat/multilingual-e5-base`.

---

## Known limitations

- 200 Wikipedia benchmark items; 68 generation evaluations; 7 retrieval-failure
  cases. Behaviour on retrieval failure is **unresolved** — two runs disagreed.
- One dialect (Moroccan Darija), one domain (Morocco-related Wikipedia), one
  generator (Qwen2.5-7B-Instruct).
- The 80 pilot passages are author-written. Questions written while reading
  their passage share 0.561 of their tokens with it, against 0.360 for an
  independent rewrite, so **pilot results carry an authorship confound**. This
  is documented in the experimental report and is why the Wikipedia subset is
  treated as primary throughout.

---

## Licensing

**Dual-licensed — see [`LICENSE.md`](LICENSE.md) before redistributing.**

Wikipedia-derived passages are **CC BY-SA 4.0** (attribution and share-alike
required). All original work — QA pairs, notebooks, documentation — is
**Apache 2.0**.

---

## Citation

Paper in preparation. Please cite this repository in the interim.
