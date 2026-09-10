# Datasheet: MSA–Darija Parallel QA Benchmark

Following the spirit of Gebru et al., "Datasheets for Datasets" (2018).

## Motivation

**For what purpose was the dataset created?**
To provide a controlled, parallel benchmark for measuring how much a
dialectal query (Moroccan Darija) degrades retrieval and generation
faithfulness in an Arabic RAG system, compared to a Modern Standard Arabic
(MSA) query, against the *same* MSA knowledge base. Existing Arabic RAG
evaluations are typically MSA-only, so this failure mode is normally
invisible in reported results.

**Who created this dataset?**
Built as part of the "Dialect-Aware Arabic Retrieval-Augmented Generation"
project, targeting IEEE-ICCA 2026. Items were AI-assisted drafts, reviewed
and corrected by the project author (a native Moroccan Darija speaker).

## Composition

**What do the instances represent?**
Each instance is a QA triple: a question in MSA, the same question in
Moroccan Darija, a gold answer, and a pointer to the MSA source passage
(`source_chunk_id`) that grounds the answer.

**How many instances are there?**
300 QA pairs, grounded in 80 MSA reference passages (avg. 3.75 items per
passage, range 1–5).

**What topics are covered?**
- History and geography (e.g. cities, founding dates, coastline)
- Social and economic indicators (e.g. unemployment, minimum wage, tourism,
  remittances, social programs)
- Institutions and administrative procedures (e.g. ID cards, passports,
  driving licenses, civil registry, bank accounts)
- Daily life and conversational topics (e.g. grocery shopping, taxis,
  the souk, pharmacies, cafés, Ramadan, weddings, family customs, weather)

**Is any information missing from individual instances?**
No — every instance has all five required fields (`id`, `msa_query`,
`darija_query`, `gold_answer`, `source_chunk_id`), enforced by an automated
schema check before commit.

**Are relationships between instances made explicit?**
Yes, via `source_chunk_id`: multiple QA items can share the same source
passage, and `corpus.json` provides the passage text.

## Collection process

**How was the data collected/created?**
1. A synthetic MSA reference corpus (80 short passages) was written to cover
   the target topic areas.
2. QA items were AI-drafted in batches (MSA question, Darija translation,
   grounded answer) against these passages.
3. Each batch was passed through an automated 6-check validation pipeline
   (see `validation/`).
4. Flagged items were reviewed and corrected by a native Darija speaker
   before being merged into the final dataset.

**Over what timeframe was the data collected?**
Built iteratively in batches (34 → 100 → 200 → 300 items) during the
project's Week 2 dataset-construction phase.

## Preprocessing / cleaning / labeling

**Was any preprocessing/cleaning/labeling done?**
Yes — the automated pipeline performed:
- Schema and referential-integrity validation (Pandera)
- MSA/Darija semantic alignment scoring (multilingual sentence embeddings)
- Dialect-authenticity lexicon scan (Darija-specific marker words)
- Answer-grounding verification via an LLM-as-judge (Groq, `openai/gpt-oss-120b`
  / `openai/gpt-oss-20b`)
- Near-duplicate detection (embedding similarity)
- Corpus coverage check (usage distribution across passages)

Flagged items (~35–40% across build batches, consistent with expectations for
an AI-assisted-then-human-reviewed pipeline) were manually reviewed and
corrected; the final 300-item set has been reviewed and confirmed by the
native-speaker author.

**Is the raw/unprocessed data also available?**
Batch-level flagged-item review files are kept under
`validation/flagged_history/` for provenance and reproducibility of the
review process.

## Uses

**What other tasks could the dataset be used for?**
- General MSA↔Darija translation/paraphrase pair research
- Dialect-authenticity or dialect-identification classifier training
- Arabic dense-retrieval evaluation under lexical/dialectal mismatch,
  independent of the RAG use case

**Are there tasks for which the dataset should not be used?**
The corpus passages are illustrative/synthetic (written for this project,
not scraped from an authoritative source), so this dataset should not be
used as a factual reference for real-world statistics (e.g. exact minimum
wage or population figures) — those figures are simplified examples, not
verified current data.

## Distribution

**Will the dataset be publicly distributed?**
Yes, released alongside the associated paper for reproducibility.

## Maintenance

**Who maintains the dataset?**
The project author. Corrections/issues can be tracked via the repository's
issue tracker once published.
