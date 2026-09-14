# Dialect-Aware Arabic RAG — Experimental Report

**Target venue:** MDPI *Information*

**Scope of this report:** what was built, what was measured, and what the
results support. Findings are labelled **established** (significant, replicated),
**preliminary** (small sample), or **null** (tested, no effect found).

---

## 1. Research question

Arabic knowledge bases are written in Modern Standard Arabic (MSA); users ask
questions in dialect. In a Retrieval-Augmented Generation (RAG) system this
creates a hidden failure point at the *retrieval* stage: if the retriever fails
to match a dialectal query to the correct MSA passage, the generator is grounded
in the wrong context.

Three questions were investigated:

1. Does a dialectal query measurably degrade **retrieval** against an MSA corpus?
2. Does that degradation reach the **generated answers**?
3. Which **mitigation** recovers the loss?

The short answer: yes to (1), no to (2) under a well-prompted generator, and
none of the query-side mitigations tested.

Moroccan Darija was chosen because its distance from MSA is the largest among
major Arabic dialects, making the effect clearest.

---

## 2. Positioning against existing work

The closest prior work is by Maha Jarallah Althobaiti: the **ArDQA** parallel
dialectal QA benchmark (preprint, under review at *Language Resources and
Evaluation*, not publicly released), and a 2026 paper in *Information* using it.

Both address **extractive QA** — a model is given one passage and locates the
answer span within it. Retrieval is not part of that task.

**This project measures the retrieval stage**: finding the correct passage among
thousands, and the downstream effect on generated answers. That is the
distinction on which novelty rests, and it must be stated explicitly in the
paper rather than assumed.

---

## 3. What was built

### 3.1 Corpus (3,054 passages)

| Source | Passages | Provenance |
|---|---|---|
| Arabic Wikipedia | 2,974 | 519 Morocco-related articles across 12 categories, chunked to 300–900 characters, deduplicated |
| Pilot (author-written) | 80 | Written during the pilot phase |

Wikipedia text is CC BY-SA 4.0; the repository licence must reflect this.

### 3.2 Benchmark (500 parallel QA items)

| Set | Items | Grounded in | Construction |
|---|---|---|---|
| Pilot | 300 | 80 author-written passages | AI-assisted, validated, native-speaker reviewed |
| Wikipedia | 200 | Real Wikipedia passages | Hand-written against real text, verified for reference validity, dialect markers, and verbatim number grounding |

Each item: MSA question, Darija question, gold answer, source passage id.

### 3.3 Pipelines

Six-check automated validation (schema, embedding alignment, dialect-authenticity
lexicon, LLM-as-judge grounding, near-duplicate detection, coverage);
hybrid retrieval (BM25 over Arabic-normalised tokens + dense FAISS, weighted by
α); contrastive fine-tuning; and an end-to-end generation and judging pipeline.

---

## 4. Results

### 4.1 The retrieval gap — **established**

Best configuration (multilingual-e5-base, α = 0.8), Wikipedia subset,
3,054-passage corpus:

| Metric | MSA query | Darija query | Gap | 95% CI |
|---|---|---|---|---|
| Recall@1 | 0.706 | 0.559 | 0.147 | [0.090, 0.185] |
| Recall@5 | 0.923 | 0.889 | 0.034 | [0.005, 0.085] |
| MRR | 0.798 | 0.698 | 0.100 | [0.062, 0.131] |

Significant across every encoder tested and every metric. The effect concentrates
at Recall@1: the correct passage is usually still retrieved, but it is no longer
ranked first.

### 4.2 The gap does **not** significantly reach the generated answers — **null**

68 held-out Wikipedia questions, top-5 passages supplied to the generator
(Qwen2.5-7B-Instruct, 4-bit, with an explicit Arabic-only instruction):

| Condition | Gold in context | Faithfulness | **Correctness** | Refusal | Code-switching |
|---|---|---|---|---|---|
| C1 — MSA query | 0.941 | 0.926 | **0.926** | 0.015 | 0.000 |
| C2 — Darija query | 0.897 | 0.912 | **0.882** | 0.088 | 0.000 |
| C3 — Darija + fine-tuned retriever | 0.868 | 0.853 | 0.838 | 0.088 | 0.015 |
| C4 — gold passage given (oracle) | 1.000 | 0.971 | 0.941 | 0.029 | 0.015 |

| Comparison | Correctness difference | 95% CI | Significant |
|---|---|---|---|
| C1 − C2 (does dialect reach answers?) | +0.044 | [−0.029, +0.118] | no |
| C3 − C2 (does fine-tuning help answers?) | −0.044 | [−0.132, +0.044] | no |
| C4 − C2 (retrieval vs generation loss) | +0.059 | [−0.015, +0.133] | no |

**The retrieval gap does not propagate to answer quality at top-5.** Dialectal
correctness is 0.882 against 0.926 for MSA, and the difference is not
statistically distinguishable from zero.

This is internally consistent with the retrieval results. The dialect penalty
concentrates at Recall@1 (0.147), while the Recall@5 gap is only 0.034 — and the
generator receives the top five passages. The correct passage is nearly always
present; it is simply no longer ranked first, which the generator does not care
about.

### 4.3 Generator prompting matters more than retrieval mitigation — **established**

An initial run used a generation prompt without an explicit language constraint.
Qwen2.5-7B then code-switched into English on some dialectal queries
(e.g. answering *"Until April 1, 1998."*) and emitted occasional corrupted
tokens. Adding one instruction — *answer in Arabic only, no Latin characters* —
changed the outcome substantially:

| | Unconstrained prompt | Arabic-only prompt |
|---|---|---|
| C2 correctness (dialectal) | 0.824 | **0.882** |
| C1 − C2 gap | +0.132, **significant** | +0.044, **not significant** |
| Code-switching rate | not measured | 0.000 |

A single prompt constraint recovered more dialectal answer quality than any
retrieval-side mitigation tested in this project. The apparent end-to-end dialect
gap in the first run was substantially an artefact of the generator mishandling
dialectal input, not of retrieval failure.

**Implication for practitioners:** in dialectal Arabic RAG, generator prompting
deserves at least as much attention as retrieval quality. This is a concrete,
low-cost intervention, unlike encoder fine-tuning.

### 4.3b Behaviour when retrieval fails — **inconclusive**

| Gold passage | Cases | Refusal rate | Correctness |
|---|---|---|---|
| Missing from context | 7 | 0.143 | 0.429 |
| Present | 61 | 0.082 | 0.934 |

Under the unconstrained prompt the same 7 cases produced a refusal rate of 0.714;
under the Arabic-only prompt it is 0.143. **Seven cases cannot support either
conclusion**, and the two runs disagree sharply. No claim should be made about
whether the generator refuses safely or answers confidently from wrong context
without a substantially larger sample of retrieval failures.

### 4.4 Query-side mitigation does not work — **null, with diagnosis**

Four strategies tested: LLM normalisation (two prompt designs), rule-based
lexicon substitution, and query expansion — across four encoders and both
subsets. **No configuration produced a statistically significant improvement
over the raw dialectal query.**

The diagnosis was reached by falsifying three candidate explanations and
supporting a fourth:

| Hypothesis | Test | Outcome |
|---|---|---|
| Co-authored pilot corpus inflates results | Rerun on 3,054 passages | Falsified |
| Normalisation destroys lexical overlap | Token overlap with gold | Falsified (0.360 vs 0.309 — *higher*) |
| Normalisation destroys discriminative terms | IDF-weighted overlap | Falsified (0.341 vs 0.242 — *higher*) |
| Normalisation drifts toward Wikipedia register | Source of retrieved passages | Falsified (77.0% vs 76.7%) |
| **Normalisation reduces ranking discriminability** | Gold score minus best competitor | **Supported** |

| Query form | Gold-vs-competitor margin | Recall@5 |
|---|---|---|
| MSA | +0.2737 | 0.913 |
| Darija | +0.0397 | 0.767 |
| Normalised | +0.0086 | 0.697 |

**Interpretation.** A fluent MSA question matches its target passage well *and
also matches many near-neighbour passages well*, collapsing the margin that
determines rank. A dialectal query is lexically unusual: it matches fewer
passages, but retains enough distinctive terms that the gold stays ahead by a
wider margin. Replacing a dialectal query with a fluent paraphrase can therefore
reduce ranking quality while improving semantic fidelity.

### 4.5 Encoder comparison — **established**

| Encoder | MSA R@5 | Darija R@5 | Gap |
|---|---|---|---|
| multilingual-e5-base | 0.950 | 0.905 | **0.045** |
| GATE-AraBert-v1 | 0.910 | 0.855 | 0.055 |
| Arabert-all-nli-Matryoshka | 0.940 | 0.850 | 0.090 |
| Arabic-Triplet-Matryoshka-V2 | 0.930 | 0.835 | 0.095 |

A general multilingual model outperformed every Arabic-specialised model tested,
including GATE. Separately, `paraphrase-multilingual-MiniLM-L12-v2` was found to
be **actively harmful**: retrieval quality fell monotonically as dense weighting
increased (Darija R@5 0.767 at α = 0.3 → 0.367 at α = 1.0).

### 4.6 Contrastive fine-tuning — **established for retrieval, null end-to-end**

Fine-tuning the encoder on the parallel MSA/Darija pairs, trained and evaluated
on Wikipedia data only, with passage-level splits and five random seeds:

| Metric | Darija before | Darija after | Gain | Seeds positive |
|---|---|---|---|---|
| Recall@1 | 0.559 | 0.630 | **+0.070** | 5/5 |
| MRR | 0.698 | 0.741 | **+0.044** | 5/5 |
| Recall@5 | 0.889 | 0.897 | +0.008 | 3/5 |

Stable across 2, 3 and 5 epochs. MSA performance did not regress (+0.036 R@1),
so the improvement is not a trade-off. **23.1% of the Recall@1 gap closed.**

**But end-to-end correctness did not improve** (C3 − C2 = +0.000,
95% CI [−0.088, +0.074]). This is internally consistent: the gains are at
Recall@1 and MRR while Recall@5 is flat, and the generator receives the top-5.
**Reordering within the top-k does not change what the generator sees.**

### 4.7 A methodological finding — **established**

An earlier fine-tuning run using mixed pilot and Wikipedia data appeared much
stronger (+0.110 Recall@5). Per-subset analysis showed the gain was confined to
the pilot subset:

| Subset | R@1 gain | R@5 gain | Seeds positive (R@5) |
|---|---|---|---|
| Pilot | +0.138 | +0.110 | 5/5 |
| Wikipedia | +0.034 | −0.002 | 1/5 |

The split was by source passage, which prevents memorising individual passages —
but not the **author's writing style**, shared across all 80 pilot passages.
Removing pilot data entirely *doubled* the Wikipedia effect (+0.034 → +0.070).

Separately, MSA questions written while reading their passages share 0.561 of
their tokens with them, against 0.360 for an independent rewrite — so
author-written benchmarks inflate measured baselines.

---

## 5. Summary of findings

| # | Finding | Status |
|---|---|---|
| 1 | Dialectal queries reduce Recall@1 by 14.7 points on a 3,054-passage corpus | Established |
| 2 | The gap does **not** significantly reach generated answers at top-5 (0.926 vs 0.882) | Null |
| 3 | Query-side mitigation fails across 4 encoders × 4 strategies | Null, with diagnosis |
| 4 | Cause: normalisation collapses the gold-vs-competitor ranking margin | Established |
| 5 | Contrastive fine-tuning closes 23% of the Recall@1 gap, no MSA regression | Established |
| 6 | Recall@1 gains do not improve RAG answers when the generator sees top-k | Established |
| 10 | Behaviour on retrieval failure is unresolved (n=7, runs disagree) | Inconclusive |
| 7 | Generator prompting recovered more dialectal answer quality than any retrieval mitigation | Established |
| 8 | General multilingual encoders outperform Arabic-specialised ones here | Established |
| 9 | Author-written benchmarks inflate results via shared writing style | Established |

---

## 6. Limitations

- **Sample size.** 200 Wikipedia items; 68 generation evaluations; 7
  retrieval-failure cases. The refusal finding in particular is preliminary.
- **Single dialect, single corpus, single domain.** Moroccan Darija over
  Morocco-related Wikipedia. Generalisation untested.
- **Benchmark construction.** Items were AI-assisted or hand-written by the
  author, then validated. The 80 pilot passages are author-written and their
  confound is documented in §4.7.
- **Single generator.** All generation results use Qwen2.5-7B-Instruct in 4-bit.
  A second generator (e.g. Atlas-Chat, Darija-specialised) would test whether
  the top-5 robustness in §4.2 is generator-specific. Not yet run.
- **Prompt sensitivity.** §4.3 shows results move substantially with one prompt
  instruction, so any single-prompt generation result should be read cautiously.
- **Partial mitigation.** Fine-tuning closes 23% of the Recall@1 gap; the gap
  remains significant.

---

## 7. Remaining work

1. **Enlarge the generation evaluation.** Because the fine-tuned condition (C3)
   showed no effect, the train/test split it required can be dropped, raising the
   generation sample from 68 to all 200 Wikipedia items at no extra data cost.
   This is the cheapest available improvement and directly addresses the widest
   confidence intervals in §4.2.
2. **Second generator.** Repeat §4.2 with a Darija-specialised model
   (Atlas-Chat) to establish whether top-5 robustness generalises or is specific
   to Qwen.
3. **More retrieval-failure cases.** §4.3b is unresolved at n = 7. Either enlarge
   the sample or construct a condition with lower retrieval success (for example
   top-1 instead of top-5) so failures are frequent enough to analyse.
4. **Optional:** expand the Wikipedia benchmark beyond 200 items to tighten all
   confidence intervals.
5. **Repository:** commit the current corpus and benchmark; apply dual licensing
   (CC BY-SA 4.0 for Wikipedia-derived passages, permissive for original QA pairs
   and code); update the datasheet with the §4.7 confound.

## 8. Assessment

The study establishes a clear, replicated dialect penalty in **retrieval**:
14.7 points of Recall@1 on a 3,054-passage corpus, significant across every
encoder tested. It further shows that the obvious remedy — normalising the
dialectal query into MSA — does not work under any of four strategies and four
encoders, and it identifies the mechanism: normalisation collapses the margin
between the correct passage and its nearest competitors.

The end-to-end picture is more nuanced than the project originally assumed. Once
the generator is instructed to answer in Arabic, the retrieval penalty does
**not** translate into significantly worse answers at top-5, because the dialect
effect is concentrated at rank 1 while the generator reads the top five
passages. The practical conclusion is that a RAG pipeline absorbs a moderate
ranking penalty, and that generator prompting is a more effective lever than
retrieval-side mitigation in this setting.

Three results are non-obvious and would be of interest to the field: the
refutation of query normalisation together with its mechanistic explanation; the
finding that Recall@1 improvements do not improve RAG answers when the generator
receives top-k; and the demonstration that author-written benchmarks inflate
measured baselines through shared writing style, which passage-level splitting
does not prevent.

The principal weakness is scale: 200 benchmark items, 68 generation evaluations,
7 retrieval-failure cases, one dialect, one domain, one generator. Item 1 of §7
addresses the most consequential of these at negligible cost and should be
completed before submission.
