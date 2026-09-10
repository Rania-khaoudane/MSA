# MSA–Darija Parallel QA Benchmark

A parallel Modern Standard Arabic (MSA) / Moroccan Darija question-answering
benchmark, built for the **Dialect-Aware Arabic Retrieval-Augmented Generation**
project (target venue: IEEE-ICCA 2026).

This dataset is the C1 contribution of the project: a controlled benchmark in
which each question exists in two aligned forms — an MSA version and a Darija
version — over the same MSA knowledge base, grounded in real passage text.
It is designed to let a RAG system be evaluated on retrieval and generation
faithfulness under two conditions (MSA query vs. Darija query) against the
identical corpus, isolating the effect of dialectal mismatch.

## Contents

```
data/
  corpus.json           80 MSA reference passages (id + text)
  qa_pairs.json          300 QA items (id, msa_query, darija_query, gold_answer, source_chunk_id)
  qa_pairs_merged.json   same 300 items, with source passage text joined in for convenience
  stats.json              basic dataset statistics
validation/
  dataset_validation.ipynb   the automated validation notebook used to build this dataset
  flagged_history/            record of items flagged and reviewed across build batches
DATASHEET.md            full datasheet (motivation, composition, collection, uses, etc.)
LICENSE
```

## Quick stats

| | |
|---|---|
| QA pairs | 300 |
| Corpus passages | 80 |
| Avg. items per passage | 3.75 |
| Languages | Modern Standard Arabic (MSA), Moroccan Darija |
| Topics | History, geography, social/economic issues, and everyday/daily-life conversation (shopping, services, family, religion, health, education, etc.) |

## How this dataset was built

Items were drafted with AI assistance against the 80-passage MSA corpus, then
run through a 6-check automated validation pipeline (schema/referential
integrity, MSA–Darija semantic alignment, dialect-authenticity lexicon scan,
LLM-as-judge answer grounding, near-duplicate detection, and corpus coverage),
and reviewed by a native Moroccan Darija speaker. See `DATASHEET.md` for full
detail and `validation/` for the exact tooling used.

## Using this dataset

```python
import json

corpus = json.load(open("data/corpus.json", encoding="utf-8"))
qa_pairs = json.load(open("data/qa_pairs.json", encoding="utf-8"))

corpus_map = {c["chunk_id"]: c["text"] for c in corpus}

# Example: matched (baseline) vs. mismatched (Darija) query for the same item
item = qa_pairs[0]
print("MSA query:   ", item["msa_query"])
print("Darija query:", item["darija_query"])
print("Gold answer: ", item["gold_answer"])
print("Source text: ", corpus_map[item["source_chunk_id"]])
```

For a RAG retrieval experiment, pool all passages in `corpus.json` into a
single searchable index and query it with either `msa_query` (baseline
condition) or `darija_query` (mismatch condition), using `source_chunk_id` as
the retrieval ground truth for Recall@k / MRR.

## License

See `LICENSE`. (Default suggestion: CC-BY-4.0 — update once finalized.)

## Citation

If you use this dataset, please cite the associated paper (details to be
added once published at IEEE-ICCA 2026).
