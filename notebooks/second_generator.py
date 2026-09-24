# =============================================================================
# Point K: do the conclusions hold with a different generator?
#
# Every end-to-end number currently comes from one model, Qwen2.5-7B-Instruct.
# The reviewer asks whether the findings are a property of the pipeline or of
# that model. This reruns generation with a second generator over the same
# questions, the same retrieval results, and the same judge, so the generator
# is the only thing that changes.
#
# Atlas-Chat-9B is the default second model: it is Gemma-2 based rather than
# Qwen, and is instruction-tuned for Moroccan Darija, which makes it the
# strongest available test of whether a Darija-aware generator behaves
# differently on dialectal input.
#
# Retrieval, reranking and gating are reused from rag_end_to_end.py's
# checkpoint; nothing is re-retrieved.
#
# RUN with a GPU, after rag_end_to_end.py. Checkpointed.
# =============================================================================

CONFIG = {
    "corpus_file": "corpus_v2.json",
    "qa_file": "qa_pairs_wiki.json",
    "retrieval_ckpt": "rag_outputs/retrieval.json",
    "generator": "MBZUAI-Paris/Atlas-Chat-9B",
    "fallback_generators": ["MBZUAI-Paris/Atlas-Chat-2B", "Qwen/Qwen2.5-3B-Instruct"],
    "judge": "Qwen/Qwen2.5-14B-Instruct",
    # The conditions that carry the paper's claims. Add more at proportional cost.
    "conditions": ["A", "B", "C", "F", "H"],
    "gen_batch": 4,
    "judge_batch": 4,
    "max_new_tokens": 96,
    "seed": 42,
    "bootstrap_n": 4000,
    "out": "second_generator_outputs",
    # Comparison baseline: the first generator's results, to test whether the
    # same comparisons reach the same conclusions.
    "baseline_raw": "rag_outputs/rag_generation_raw.csv",
}

import os, re, json, gc
import numpy as np
import pandas as pd
from tqdm import tqdm

OUT = CONFIG["out"]
os.makedirs(OUT, exist_ok=True)

corpus = json.load(open(CONFIG["corpus_file"], encoding="utf-8"))
qa = [q for q in json.load(open(CONFIG["qa_file"], encoding="utf-8"))]
ids = [c["chunk_id"] for c in corpus]
texts = [c["text"] for c in corpus]
pos = {c: i for i, c in enumerate(ids)}
qa = [q for q in qa if q["source_chunk_id"] in pos]
byid = {q["id"]: q for q in qa}
retr = json.load(open(CONFIG["retrieval_ckpt"], encoding="utf-8"))
print(f"{len(qa)} questions | reusing retrieval conditions: {list(retr)}")

COND = {"A": ("MSA", 5, "msa_query"), "B": ("Darija", 5, "darija_query"),
        "C": ("Darija", 1, "darija_query"), "D": ("Darija-norm", 5, "darija_query"),
        "E": ("Darija-rerank", 5, "darija_query"), "F": ("Darija-rerank", 1, "darija_query"),
        "G": ("Darija-gated", 1, "darija_query"), "H": ("oracle", 1, "darija_query")}
LABEL = {"A": "MSA · k=5", "B": "Darija · k=5", "C": "Darija · k=1",
         "D": "Darija norm · k=5", "E": "Darija rerank · k=5",
         "F": "Darija rerank · k=1", "G": "Darija gated · k=1", "H": "oracle"}

GEN_PROMPT = """أجب عن السؤال التالي اعتمادا فقط على النصوص المرفقة.

قواعد إلزامية:
- أجب بالعربية فقط. ممنوع استعمال أي كلمة بحرف لاتيني.
- إذا لم تكن الإجابة موجودة في النصوص، اكتب بالضبط: المعلومة غير متوفرة في النصوص
- لا تستعمل أي معرفة خارجية.
- أجب بجملة واحدة قصيرة فقط.

النصوص:
{context}

السؤال: {question}

الإجابة:"""

JUDGE_PROMPT = """النصوص المرجعية:
{context}

السؤال: {question}
الإجابة الصحيحة: {gold}
الإجابة المقدمة: {answer}

قيّم الإجابة المقدمة وفق القواعد التالية:

قاعدة الامتناع: إذا قالت الإجابة المقدمة إن المعلومة غير متوفرة، فالجواب: مدعوم: لا، مطابق: لا.

قاعدة المطابقة:
- يكفي أن تتضمن الإجابة المقدمة الجوهر الصحيح؛ لا يُشترط ذكر كل التفاصيل.
- إضافة تفاصيل صحيحة إضافية لا تُبطل المطابقة.
- اختلاف الأرقام أو الأسماء أو التواريخ يعني: مطابق: لا.

أجب بهذا الشكل فقط وبدون أي شرح:
مدعوم: نعم/لا
مطابق: نعم/لا"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
llm = tokenizer = None
def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_llm(name):
    global llm, tokenizer
    if llm is not None:
        del llm; llm = None; free()
    q4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    llm = AutoModelForCausalLM.from_pretrained(name, quantization_config=q4,
                                               device_map="auto", torch_dtype=torch.float16).eval()
    print("loaded", name)

def load_with_fallback(primary, fallbacks):
    for name in [primary] + list(fallbacks):
        try:
            load_llm(name)
            return name
        except Exception as e:
            print(f"  could not load {name}: {type(e).__name__}: {str(e)[:140]}")
    raise SystemExit("No generator could be loaded.")

@torch.no_grad()
def chat_batch(prompts, mnt=96):
    t = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True) for p in prompts]
    e = tokenizer(t, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(llm.device)
    o = llm.generate(**e, max_new_tokens=mnt, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return [tokenizer.decode(g, skip_special_tokens=True).strip() for g in o[:, e["input_ids"].shape[1]:]]

def context_ids(c, qid):
    name, k, _ = COND[c]
    return [byid[qid]["source_chunk_id"]] if name == "oracle" else retr[name][qid]["cands"][:k]
def ctx_text(cids):
    return "\n\n".join(f"[{i+1}] {texts[pos[c]]}" for i, c in enumerate(cids))

CKPT = os.path.join(OUT, "generations.json")
gen = json.load(open(CKPT, encoding="utf-8")) if os.path.exists(CKPT) else []
def save(): json.dump(gen, open(CKPT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

used_generator = load_with_fallback(CONFIG["generator"], CONFIG["fallback_generators"])
done = {(g["qid"], g["cond"]) for g in gen}
for c in CONFIG["conditions"]:
    todo = [q for q in qa if (q["id"], c) not in done]
    if not todo:
        continue
    print(f"\ngenerating {c}: {LABEL[c]} ({len(todo)} left)")
    for i in tqdm(range(0, len(todo), CONFIG["gen_batch"]), desc=f"gen {c}"):
        b = todo[i:i + CONFIG["gen_batch"]]
        cids = [context_ids(c, q["id"]) for q in b]
        outs = chat_batch([GEN_PROMPT.format(context=ctx_text(x), question=q[COND[c][2]])
                           for q, x in zip(b, cids)], CONFIG["max_new_tokens"])
        for q, x, a in zip(b, cids, outs):
            gen.append({"qid": q["id"], "cond": c, "context": x, "answer": a,
                        "gold_in_context": int(q["source_chunk_id"] in x),
                        "generator": used_generator})
        save()

load_llm(CONFIG["judge"])
def parse(v):
    t = (v or "").replace("،", " ")
    f = c = None
    for line in t.split("\n"):
        if "مدعوم" in line and f is None:
            f = int("نعم" in line.split("مطابق")[0])
        if "مطابق" in line:
            c = int("نعم" in line.split("مطابق", 1)[1])
    return (f or 0), (c or 0), (f is not None and c is not None)

todo = [g for g in gen if "correct" not in g]
for i in tqdm(range(0, len(todo), CONFIG["judge_batch"]), desc="judging"):
    b = todo[i:i + CONFIG["judge_batch"]]
    vs = chat_batch([JUDGE_PROMPT.format(context=ctx_text(g["context"]),
                                         question=byid[g["qid"]]["msa_query"],
                                         gold=byid[g["qid"]]["gold_answer"],
                                         answer=g["answer"]) for g in b], 32)
    for g, v in zip(b, vs):
        g["faithful"], g["correct"], ok = parse(v)
        g["judge_parsed"], g["judge_raw"] = int(ok), v
    save()

df = pd.DataFrame(gen)
df["refused"] = df.answer.fillna("").str.contains("غير متوفرة").astype(int)
df.to_csv(os.path.join(OUT, "second_generator_raw.csv"), index=False)

order = [c for c in COND if c in set(df.cond)]
summ = df.groupby("cond").agg(n=("qid", "count"), gold_in_context=("gold_in_context", "mean"),
                              correct=("correct", "mean"), faithful=("faithful", "mean"),
                              refusal=("refused", "mean")).reindex(order)
summ.insert(0, "condition", [LABEL[c] for c in summ.index])
summ.to_csv(os.path.join(OUT, "second_generator_summary.csv"))
print("\n" + "=" * 78)
print(f"RESULTS WITH {used_generator}")
print("=" * 78)
print(summ.to_string(float_format=lambda x: f"{x:.3f}"))

rng = np.random.default_rng(CONFIG["seed"])
def paired(d):
    d = np.asarray(d, float)
    m = d[rng.integers(0, len(d), size=(CONFIG["bootstrap_n"], len(d)))].mean(1)
    return d.mean(), *np.percentile(m, [2.5, 97.5])
def compare(frame, a, b, col="correct"):
    x = frame[frame.cond == a].set_index("qid")[col]; y = frame[frame.cond == b].set_index("qid")[col]
    i = x.index.intersection(y.index)
    return paired((x.loc[i] - y.loc[i]).values)

PAIRS = [("B", "A", "Dialect gap in answers"), ("F", "C", "Reranking at k=1"),
         ("C", "B", "Cost of k=1 without reranking"), ("H", "F", "Headroom to the oracle")]
base = pd.read_csv(CONFIG["baseline_raw"]) if os.path.exists(CONFIG["baseline_raw"]) else None

print("\n" + "=" * 78)
print("DO THE CONCLUSIONS HOLD ACROSS BOTH GENERATORS?")
print("=" * 78)
rows = []
for a, b, lab in PAIRS:
    if a not in order or b not in order:
        continue
    d2, lo2, hi2 = compare(df, a, b)
    sig2 = "yes" if lo2 > 0 or hi2 < 0 else "no"
    row = {"comparison": lab, "gen2_diff": d2, "gen2_lo": lo2, "gen2_hi": hi2, "gen2_sig": sig2}
    if base is not None and {a, b} <= set(base.cond):
        d1, lo1, hi1 = compare(base, a, b)
        sig1 = "yes" if lo1 > 0 or hi1 < 0 else "no"
        row.update({"gen1_diff": d1, "gen1_lo": lo1, "gen1_hi": hi1, "gen1_sig": sig1,
                    "same_sign": (d1 > 0) == (d2 > 0), "same_significance": sig1 == sig2})
    rows.append(row)
C = pd.DataFrame(rows)
C.to_csv(os.path.join(OUT, "generator_agreement.csv"), index=False)
print(C.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

if "same_sign" in C.columns:
    n_sign = int(C.same_sign.sum()); n_sig = int(C.same_significance.sum())
    print(f"""
  {n_sign}/{len(C)} comparisons keep the same direction across generators.
  {n_sig}/{len(C)} keep the same significance.

  Conclusions that hold for both generators can be stated as findings about the
  pipeline. Any that differ must be reported as generator-specific, with both
  values given.""")
print(f"\nWrote {OUT}/")
