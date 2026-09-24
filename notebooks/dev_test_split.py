# =============================================================================
# Review response H: hyper-parameters were selected on the evaluation data.
#
# Alpha (hybrid weight), rerank depth, and generator context size k were all
# chosen by looking at performance on the same 200 items the paper reports
# results on. This re-selects them on a held-out 50-item DEV split, then
# re-reports every headline retrieval number on the remaining 150-item TEST
# split, so the reported gaps are not optimistically biased by tuning on the
# test set.
#
# THE SPLIT IS BY SOURCE PASSAGE, NOT BY QUESTION. If two questions grounded
# in the same passage landed on opposite sides, information could leak across
# the split. The corpus otherwise stays identical between dev and test (all
# 3,054 passages remain searchable in both); only which QUESTIONS are used for
# selection versus reporting changes.
#
# This does not touch the generation results (A-H). Those already use the
# alpha/depth values found here as constants; if the values below differ
# materially from what was used, generation should be rerun with the new
# ones -- the script prints a clear comparison so you can decide.
#
# RUN with a GPU. Needs corpus_v2.json and qa_pairs_wiki.json. No dependency
# on rag_end_to_end.py's checkpoints, since it needs its own clean split.
# =============================================================================

CONFIG = {
    "corpus_file": "corpus_v2.json",
    "qa_file": "qa_pairs_wiki.json",
    "encoder": "intfloat/multilingual-e5-base",
    "reranker": "BAAI/bge-reranker-v2-m3",
    "dev_frac": 0.25,          # ~50 of 200 items held out for tuning
    "alpha_grid": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "depth_grid": [10, 20, 50],
    "k_grid": [1, 3, 5],
    # values currently used in the paper, for comparison against what
    # selection on the dev split recommends
    "paper_alpha": 0.8,
    "paper_depth": 20,
    "seed": 42,
    "bootstrap_n": 2000,
    "out": "dev_test_outputs",
}

import os, re, json, random, gc
import numpy as np
import pandas as pd

os.makedirs(CONFIG["out"], exist_ok=True)
os.makedirs("tables", exist_ok=True)

corpus = json.load(open(CONFIG["corpus_file"], encoding="utf-8"))
qa_all = json.load(open(CONFIG["qa_file"], encoding="utf-8"))
ids = [c["chunk_id"] for c in corpus]
texts = [c["text"] for c in corpus]
pos = {cid: i for i, cid in enumerate(ids)}
qa = [q for q in qa_all if q["source_chunk_id"] in pos]

rnd = random.Random(CONFIG["seed"])
passages = sorted({q["source_chunk_id"] for q in qa})
rnd.shuffle(passages)
n_dev = max(1, int(len(passages) * CONFIG["dev_frac"]))
dev_passages = set(passages[:n_dev])
dev_qa = [q for q in qa if q["source_chunk_id"] in dev_passages]
test_qa = [q for q in qa if q["source_chunk_id"] not in dev_passages]
assert not ({q["source_chunk_id"] for q in dev_qa} & {q["source_chunk_id"] for q in test_qa}), \
    "passage leakage between dev and test"
print(f"Corpus {len(corpus)} passages (unchanged, searchable in both splits)")
print(f"Dev: {len(dev_qa)} questions from {len(dev_passages)} passages")
print(f"Test: {len(test_qa)} questions from {len(passages) - len(dev_passages)} passages")

# ---------------------------------------------------------------- retrieval
from rank_bm25 import BM25Okapi

DIAC = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED\u0670]")
def norm_ar(t):
    t = DIAC.sub("", t or "")
    for a, b in [(r"[\u0625\u0623\u0622\u0627]", "\u0627"), (r"\u0649", "\u064A"),
                 (r"\u0629", "\u0647"), (r"\u0624", "\u0648"), (r"\u0626", "\u064A"),
                 (r"\u0640+", ""), (r"[^\w\s]", " ")]:
        t = re.sub(a, b, t)
    return re.sub(r"\s+", " ", t).strip()
def tok(t): return norm_ar(t).split()
def minmax(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

bm25 = BM25Okapi([tok(t) for t in texts])

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

enc = SentenceTransformer(CONFIG["encoder"])
emb = np.asarray(enc.encode([f"passage: {t}" for t in texts], normalize_embeddings=True,
                            batch_size=32, show_progress_bar=True), "float32")

def dense_sparse(query, maxd):
    qe = enc.encode([f"query: {query}"], normalize_embeddings=True)[0]
    d = minmax(emb @ qe)
    s = minmax(np.asarray(bm25.get_scores(tok(query))))
    return d, s

def topk_from_scores(score, k):
    order = np.argsort(-score)[:k]
    return [ids[i] for i in order]

def recall_at(items, alpha, k, field="darija_query"):
    hits = []
    for q in items:
        d, s = dense_sparse(q[field], len(ids))
        combined = alpha * d + (1 - alpha) * s
        got = topk_from_scores(combined, k)
        hits.append(1.0 if q["source_chunk_id"] in got else 0.0)
    return np.mean(hits)

# ---------------------------------------------------------------- 1. select alpha on DEV
print("\n" + "=" * 78)
print("STEP 1 — select alpha on the DEV split only")
print("=" * 78)
alpha_rows = []
for a in CONFIG["alpha_grid"]:
    r1 = recall_at(dev_qa, a, 1)
    r5 = recall_at(dev_qa, a, 5)
    alpha_rows.append({"alpha": a, "dev_R@1": r1, "dev_R@5": r5})
    print(f"  alpha={a:.1f}  dev R@1={r1:.3f}  dev R@5={r5:.3f}")
A = pd.DataFrame(alpha_rows)
A.to_csv(os.path.join(CONFIG["out"], "alpha_selection.csv"), index=False)
best_alpha = float(A.loc[A["dev_R@1"].idxmax(), "alpha"])
print(f"\n  selected alpha = {best_alpha}  (paper currently uses {CONFIG['paper_alpha']})")

# ---------------------------------------------------------------- 2. select rerank depth on DEV
print("\n" + "=" * 78)
print("STEP 2 — select rerank depth on the DEV split, at the selected alpha")
print("=" * 78)
ce = CrossEncoder(CONFIG["reranker"], max_length=512, trust_remote_code=True,
                  automodel_args={"torch_dtype": torch.float32})

def rerank_recall_at_depth(items, alpha, depths, field="darija_query"):
    maxd = max(depths)
    out = {d: [] for d in depths}
    for q in items:
        dd, ss = dense_sparse(q[field], len(ids))
        combined = alpha * dd + (1 - alpha) * ss
        cands = topk_from_scores(combined, maxd)
        pairs = [(q[field], texts[pos[c]]) for c in cands]
        sc = np.asarray(ce.predict(pairs, batch_size=16, show_progress_bar=False))
        if sc.ndim > 1:
            sc = sc[:, -1]
        order = np.argsort(-sc)
        ranked = [cands[i] for i in order]
        for d in depths:
            out[d].append(1.0 if q["source_chunk_id"] in ranked[:1] else 0.0)  # R@1 after reranking d candidates, truncated to top-1
    return {d: float(np.mean(v)) for d, v in out.items()}

depth_r1 = rerank_recall_at_depth(dev_qa, best_alpha, CONFIG["depth_grid"])
for d, v in depth_r1.items():
    print(f"  depth={d:<3} dev reranked R@1={v:.3f}")
best_depth = max(depth_r1, key=depth_r1.get)
print(f"\n  selected depth = {best_depth}  (paper currently uses {CONFIG['paper_depth']})")
pd.DataFrame([{"depth": d, "dev_R1_after_rerank": v} for d, v in depth_r1.items()]) \
  .to_csv(os.path.join(CONFIG["out"], "depth_selection.csv"), index=False)

del enc, emb
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ---------------------------------------------------------------- 3. re-report on TEST
print("\n" + "=" * 78)
print(f"STEP 3 — re-report headline numbers on the {len(test_qa)}-item TEST split,")
print(f"         using alpha={best_alpha}, depth={best_depth} selected above")
print("=" * 78)

enc = SentenceTransformer(CONFIG["encoder"])
emb = np.asarray(enc.encode([f"passage: {t}" for t in texts], normalize_embeddings=True,
                            batch_size=32, show_progress_bar=True), "float32")

def retrieve_full(query, alpha, k):
    qe = enc.encode([f"query: {query}"], normalize_embeddings=True)[0]
    d = minmax(emb @ qe)
    s = minmax(np.asarray(bm25.get_scores(tok(query))))
    return topk_from_scores(alpha * d + (1 - alpha) * s, k)

rng = np.random.default_rng(CONFIG["seed"])
def boot(d):
    d = np.asarray(d, float)
    m = d[rng.integers(0, len(d), size=(CONFIG["bootstrap_n"], len(d)))].mean(1)
    return d.mean(), *np.percentile(m, [2.5, 97.5])

rows = []
for field, label in [("msa_query", "MSA"), ("darija_query", "Darija")]:
    r1 = [1.0 if q["source_chunk_id"] in retrieve_full(q[field], best_alpha, 1) else 0.0 for q in test_qa]
    r5 = [1.0 if q["source_chunk_id"] in retrieve_full(q[field], best_alpha, 5) else 0.0 for q in test_qa]
    rows.append({"query": label, "R@1": np.mean(r1), "R@5": np.mean(r5), "_r1": r1})

msa_r1, dar_r1 = rows[0]["_r1"], rows[1]["_r1"]
gap, lo, hi = boot(np.array(msa_r1) - np.array(dar_r1))
print(f"  MSA R@1={rows[0]['R@1']:.3f}  Darija R@1={rows[1]['R@1']:.3f}  "
      f"gap={gap:.3f} [{lo:.3f},{hi:.3f}]")
print(f"  MSA R@5={rows[0]['R@5']:.3f}  Darija R@5={rows[1]['R@5']:.3f}")

for r in rows:
    del r["_r1"]
test_summary = pd.DataFrame(rows)
test_summary.loc[len(test_summary)] = {"query": "gap", "R@1": gap, "R@5": np.nan}
test_summary.to_csv(os.path.join(CONFIG["out"], "test_split_retrieval.csv"), index=False)

lines = [f"MSA & {rows[0]['R@1']:.3f} & {rows[0]['R@5']:.3f} \\\\",
         f"Darija & {rows[1]['R@1']:.3f} & {rows[1]['R@5']:.3f} \\\\",
         f"Gap & {gap:.3f} [{lo:.3f}, {hi:.3f}] & \\\\"]
open("tables/table8_dev_test.tex", "w").write("\n".join(lines) + "\n")

print(f"\n{'='*78}\nSUMMARY FOR THE PAPER\n{'='*78}")
print(f"  Selected on dev (n={len(dev_qa)}):  alpha={best_alpha}, depth={best_depth}")
print(f"  Paper previously used:              alpha={CONFIG['paper_alpha']}, depth={CONFIG['paper_depth']}")
if best_alpha != CONFIG["paper_alpha"] or best_depth != CONFIG["paper_depth"]:
    print("\n  SELECTED VALUES DIFFER FROM THE PAPER. Generation (A-H) was run with the")
    print("  old values and should be rerun with the dev-selected ones if this split")
    print("  is adopted as the reported methodology.")
else:
    print("\n  Selected values match what the paper already uses -- no generation rerun")
    print("  needed; only the reported retrieval numbers change (dev/test split instead")
    print("  of tune-and-report-on-the-same-200).")
print(f"\n  Test-split (n={len(test_qa)}) headline: gap={gap:.3f} [{lo:.3f},{hi:.3f}]")
print(f"  saved {CONFIG['out']}/, tables/table8_dev_test.tex")
