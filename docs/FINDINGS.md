# Dialect Mismatch in Arabic Retrieval-Augmented Generation

## Problem

Arabic knowledge bases are written in Modern Standard Arabic (MSA); users ask
questions in dialect. Existing work on this mismatch studies **extractive QA**,
where the correct passage is supplied and the model locates an answer span within
it. Retrieval is not part of that task.

This study measures the **retrieval stage** of a RAG pipeline — finding the
correct passage among thousands — and whether the resulting degradation reaches
the answers a user actually receives.

**Setup.** 3,054-passage Arabic corpus (2,974 Wikipedia passages from 519
Morocco-related articles, plus 80 pilot passages); 500 parallel QA items, each
existing as an MSA question and a Moroccan Darija question over the same corpus;
hybrid BM25 + dense retrieval; end-to-end generation with LLM-as-judge scoring.
All results carry bootstrap 95% confidence intervals.

---

## Finding 1 — Dialectal queries impose a large, robust ranking penalty

| Metric | MSA query | Darija query | Gap | 95% CI |
|---|---|---|---|---|
| Recall@1 | 0.706 | 0.559 | **0.147** | [0.090, 0.185] |
| Recall@5 | 0.923 | 0.889 | 0.034 | [0.005, 0.085] |
| MRR | 0.798 | 0.698 | 0.100 | [0.062, 0.131] |

Significant for every encoder tested. Critically, the penalty **concentrates at
rank 1**: the correct passage is usually still retrieved, it simply stops being
ranked first.

---

## Finding 2 — Query normalisation does not fix it, and we can say why

Translating the dialectal query into MSA before retrieval is the intuitive
remedy. It was tested as LLM normalisation (two prompt designs), rule-based
lexicon substitution, and query expansion, across four encoders and two data
subsets. **No configuration improved significantly over the untreated dialectal
query.**

Three candidate explanations were falsified by measurement before a fourth was
supported:

| Query form | Gold-vs-best-competitor margin | Recall@5 |
|---|---|---|
| MSA | +0.2737 | 0.913 |
| Darija | +0.0397 | 0.767 |
| Normalised | +0.0086 | 0.697 |

A fluent MSA question matches its target passage well — and also matches many
near-neighbour passages well, collapsing the margin that decides rank. A
dialectal query is lexically unusual: it matches fewer passages overall, but
retains enough distinctive terms that the correct one stays ahead by a wider
margin.

**Normalisation improves semantic fidelity while reducing ranking
discriminability.** Absolute overlap with the gold passage rises (0.309 → 0.360);
the winning margin falls by a factor of four.

---

## Finding 3 — The ranking penalty does not reach the answers

68 held-out questions, top-5 passages supplied to the generator:

| Condition | Gold in context | Correctness | Refusal |
|---|---|---|---|
| MSA query | 0.941 | 0.926 | 0.015 |
| Darija query | 0.897 | 0.882 | 0.088 |
| Gold passage supplied (oracle) | 1.000 | 0.941 | 0.029 |

**MSA − Darija correctness: +0.044, 95% CI [−0.029, +0.118] — not significant.**

This follows directly from Finding 1. The dialect penalty sits at Recall@1
(0.147) while the Recall@5 gap is only 0.034, and the generator reads the top
five passages. The correct passage is nearly always present; its rank is not
what the generator responds to.

**A RAG pipeline absorbs a moderate ranking penalty.** This is a useful negative
result: it bounds the practical severity of dialect mismatch in retrieval-based
systems.

---

## Finding 4 — Generator prompting outperforms every retrieval-side mitigation

An initial run used a generation prompt with no language constraint, and the
generator code-switched into English on dialectal input. Adding one instruction —
*answer in Arabic only* — produced this:

| | Unconstrained prompt | Arabic-only prompt |
|---|---|---|
| Dialectal correctness | 0.824 | **0.882** |
| MSA − Darija gap | +0.132, significant | +0.044, not significant |
| Code-switching rate | not measured | 0.000 |

**One prompt constraint recovered more dialectal answer quality than any
retrieval-side intervention tested.** It also shows that an apparent end-to-end
dialect gap can be an artefact of generator behaviour rather than retrieval
failure — a methodological caution for this literature.

---

## Finding 5 — Recall@1 gains do not translate into better answers

Contrastive fine-tuning of the encoder on the parallel MSA/Darija pairs, with
passage-level splits and five random seeds:

| Metric | Before | After | Gain | Seeds positive |
|---|---|---|---|---|
| Recall@1 | 0.559 | 0.630 | **+0.070** | 5/5 |
| MRR | 0.698 | 0.741 | **+0.044** | 5/5 |
| Recall@5 | 0.889 | 0.897 | +0.008 | 3/5 |

Stable across 2, 3 and 5 epochs, with no MSA regression (+0.036 Recall@1).
**23% of the Recall@1 gap closed.**

End-to-end correctness, however, did not move (+0.000, CI [−0.088, +0.074]) —
because Recall@5 was flat and the generator reads top-5.

**Retrieval metrics that reward rank-1 ordering can improve without improving a
RAG system at all.** Where the generator consumes top-k, Recall@k is the metric
that matters.

---

## Finding 6 — Author-written benchmarks inflate results through shared style

A fine-tuning run on mixed author-written and Wikipedia data appeared far
stronger (+0.110 Recall@5). Split by subset:

| Subset | Recall@1 gain | Recall@5 gain | Seeds positive |
|---|---|---|---|
| Author-written | +0.138 | +0.110 | 5/5 |
| Wikipedia | +0.034 | −0.002 | 1/5 |

The split was by source passage, which prevents memorising individual passages —
but not the **author's writing style**, common to every passage they wrote.
Removing author-written data entirely *doubled* the effect on real text
(+0.034 → +0.070).

Relatedly, questions written while reading their target passage share 0.561 of
their tokens with it, against 0.360 for an independent rewrite of the same
question.

**Passage-level splitting is insufficient when passages share an author.** This
affects any benchmark built this way, and is worth stating independently of the
dialect question.

---

## What this adds up to

Dialect mismatch produces a real and substantial **ranking** penalty in Arabic
retrieval, which the intuitive fix does not repair and which we explain
mechanistically. At the pipeline level, however, a RAG system largely absorbs it,
and the more effective lever proves to be generator prompting rather than
retrieval-side mitigation.

Four of the six findings are negative or diagnostic. Each is supported by
confidence intervals, and the fine-tuning result by five-seed replication.

**Deliverables:** a 3,054-passage Moroccan Arabic corpus, a 500-item parallel
MSA/Darija benchmark, a six-check validation pipeline, and reproducible
retrieval, fine-tuning and generation notebooks.

---

## Limitations and next steps

**Scale is the principal weakness:** 200 Wikipedia benchmark items, 68 generation
evaluations, 7 retrieval-failure cases, one dialect, one domain, one generator.
Behaviour on retrieval failure is unresolved — two runs disagreed sharply at
n = 7, and no claim is made about it.

**Highest-value next step:** the fine-tuned condition showed no end-to-end
effect, so the train/test split it required can be dropped. That raises the
generation evaluation from 68 items to all 200 at no additional data cost, and
directly narrows the widest confidence intervals in Finding 3.

**Then:** a second generator to test whether the top-5 robustness in Finding 3
generalises; and an evaluation at top-1 so retrieval failures become frequent
enough to analyse.
