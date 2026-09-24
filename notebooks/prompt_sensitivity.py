# =============================================================================
# Review response G: is the end-to-end conclusion robust to the prompt?
#
# One added instruction previously moved dialectal correctness from 0.824 to
# 0.882 and flipped the MSA-Darija gap from significant to not significant, so
# the end-to-end conclusion currently rests on a single prompt. This script
# reruns generation with three semantically equivalent Arabic-only prompts and
# reports the spread per condition, plus whether the sign and significance of
# the gap survive all three.
#
# It reuses the retrieval, reranking and gating already computed by
# rag_end_to_end.py, so no retrieval work is repeated. Only generation and
# judging run again, for the conditions listed in CONFIG["conditions"].
#
# RUN AFTER rag_end_to_end.py, in the same folder, with a GPU.
# Checkpointed: rerunning resumes.
# =============================================================================

CONFIG = {
    "corpus_file": "corpus_v2.json",
    "qa_file": "qa_pairs_wiki.json",
    "retrieval_ckpt": "rag_outputs/retrieval.json",
    "generator": "Qwen/Qwen2.5-7B-Instruct",
    "judge": "Qwen/Qwen2.5-14B-Instruct",
    # A subset is enough to test prompt robustness: the reference, the dialect
    # condition, and the proposed method. Add more at proportional cost.
    "conditions": ["A", "B", "F"],
    "gen_batch": 8,
    "judge_batch": 4,
    "max_new_tokens": 96,
    "seed": 42,
    "bootstrap_n": 2000,
    "out": "prompt_sensitivity_outputs",
}

import os, re, json, gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

OUT = CONFIG["out"]
os.makedirs(OUT, exist_ok=True)
os.makedirs("tables", exist_ok=True)

corpus = json.load(open(CONFIG["corpus_file"], encoding="utf-8"))
qa_all = json.load(open(CONFIG["qa_file"], encoding="utf-8"))
ids = [c["chunk_id"] for c in corpus]
texts = [c["text"] for c in corpus]
pos = {cid: i for i, cid in enumerate(ids)}
qa = [q for q in qa_all if q["source_chunk_id"] in pos]
byid = {q["id"]: q for q in qa}
retr = json.load(open(CONFIG["retrieval_ckpt"], encoding="utf-8"))
print(f"{len(qa)} questions | retrieval conditions: {list(retr)}")

COND = {"A": ("MSA", 5, "msa_query"), "B": ("Darija", 5, "darija_query"),
        "C": ("Darija", 1, "darija_query"), "D": ("Darija-norm", 5, "darija_query"),
        "E": ("Darija-rerank", 5, "darija_query"), "F": ("Darija-rerank", 1, "darija_query"),
        "G": ("Darija-gated", 1, "darija_query"), "H": ("oracle", 1, "darija_query")}
COND_LABEL = {"A": "MSA · k=5", "B": "Darija · k=5", "C": "Darija · k=1",
              "D": "Darija norm · k=5", "E": "Darija rerank · k=5",
              "F": "Darija rerank · k=1", "G": "Darija gated · k=1", "H": "oracle"}

# Three prompts with the same instructions in different wording. P1 is the one
# used in the main run, so its column should reproduce those numbers.
PROMPTS = {
"P1": """أجب عن السؤال التالي اعتمادا فقط على النصوص المرفقة.

قواعد إلزامية:
- أجب بالعربية فقط. ممنوع استعمال أي كلمة بحرف لاتيني.
- إذا لم تكن الإجابة موجودة في النصوص، اكتب بالضبط: المعلومة غير متوفرة في النصوص
- لا تستعمل أي معرفة خارجية.
- أجب بجملة واحدة قصيرة فقط.

النصوص:
{context}

السؤال: {question}

الإجابة:""",

"P2": """اعتمادا على النصوص التالية وحدها، أجب عن السؤال.

التعليمات:
- تكون الإجابة بالعربية حصرا، ولا تتضمن أي حرف لاتيني.
- إن لم تجد الإجابة في النصوص، اكتب هذه العبارة فقط: المعلومة غير متوفرة في النصوص
- لا تضف معلومات من خارج النصوص.
- اكتب جملة واحدة موجزة.

النصوص:
{context}

السؤال: {question}

الإجابة:""",

"P3": """النصوص المرفقة هي مصدرك الوحيد للإجابة عن السؤال أدناه.

يجب الالتزام بما يلي:
- الكتابة بالعربية فقط، دون أي كلمة بحروف لاتينية.
- عند غياب الإجابة من النصوص، تكون إجابتك: المعلومة غير متوفرة في النصوص
- عدم الاستعانة بأي معرفة خارجية.
- الاكتفاء بجملة واحدة قصيرة.

النصوص:
{context}

السؤال: {question}

الإجابة:"""}

JUDGE_PROMPT = """النصوص المرجعية:
{context}

السؤال: {question}
الإجابة الصحيحة: {gold}
الإجابة المقدمة: {answer}

أجب عن سؤالين بدقة:
1. هل كل ما ورد في الإجابة المقدمة مدعوم صراحة بالنصوص المرجعية؟
2. هل الإجابة المقدمة مطابقة في المعنى للإجابة الصحيحة؟ اختلاف الصياغة مقبول، أما اختلاف الأرقام أو الأسماء أو التواريخ فغير مقبول.

أجب بهذا الشكل فقط وبدون أي شرح:
مدعوم: نعم/لا
مطابق: نعم/لا"""

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

@torch.no_grad()
def chat_batch(prompts, max_new_tokens=96):
    t = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True) for p in prompts]
    e = tokenizer(t, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(llm.device)
    o = llm.generate(**e, max_new_tokens=max_new_tokens, do_sample=False,
                     pad_token_id=tokenizer.pad_token_id)
    return [tokenizer.decode(g, skip_special_tokens=True).strip() for g in o[:, e["input_ids"].shape[1]:]]

def context_ids(c, qid):
    name, k, _ = COND[c]
    return [byid[qid]["source_chunk_id"]] if name == "oracle" else retr[name][qid]["cands"][:k]
def ctx_text(cids):
    return "\n\n".join(f"[{i+1}] {texts[pos[c]]}" for i, c in enumerate(cids))

CKPT = os.path.join(OUT, "generations.json")
gen = json.load(open(CKPT, encoding="utf-8")) if os.path.exists(CKPT) else []
def save():
    json.dump(gen, open(CKPT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---- generation, one pass per prompt ----
load_llm(CONFIG["generator"])
done = {(g["qid"], g["cond"], g["prompt"]) for g in gen}
for pid, tpl in PROMPTS.items():
    for c in CONFIG["conditions"]:
        todo = [q for q in qa if (q["id"], c, pid) not in done]
        if not todo:
            continue
        print(f"generating {pid} / {c}: {COND_LABEL[c]} ({len(todo)} left)")
        for i in tqdm(range(0, len(todo), CONFIG["gen_batch"]), desc=f"{pid}-{c}"):
            b = todo[i:i + CONFIG["gen_batch"]]
            cids = [context_ids(c, q["id"]) for q in b]
            outs = chat_batch([tpl.format(context=ctx_text(x), question=q[COND[c][2]])
                               for q, x in zip(b, cids)], CONFIG["max_new_tokens"])
            for q, x, a in zip(b, cids, outs):
                gen.append({"qid": q["id"], "cond": c, "prompt": pid, "context": x, "answer": a,
                            "gold_in_context": int(q["source_chunk_id"] in x)})
            save()

# ---- judging ----
load_llm(CONFIG["judge"])
def parse(v):
    f = c = 0
    for line in (v or "").replace("،", " ").split("\n"):
        if "مدعوم" in line: f = int("نعم" in line)
        elif "مطابق" in line: c = int("نعم" in line)
    return f, c

todo = [g for g in gen if "correct" not in g]
for i in tqdm(range(0, len(todo), CONFIG["judge_batch"]), desc="judging"):
    b = todo[i:i + CONFIG["judge_batch"]]
    vs = chat_batch([JUDGE_PROMPT.format(context=ctx_text(g["context"]),
                                         question=byid[g["qid"]]["msa_query"],
                                         gold=byid[g["qid"]]["gold_answer"],
                                         answer=g["answer"]) for g in b], 24)
    for g, v in zip(b, vs):
        g["faithful"], g["correct"] = parse(v)
        g["judge_raw"] = v
    save()

# ---- analysis ----
df = pd.DataFrame(gen)
df["refused"] = df.answer.fillna("").str.contains("غير متوفرة").astype(int)
df.to_csv(os.path.join(OUT, "prompt_sensitivity_raw.csv"), index=False)

print("\n" + "=" * 78)
print("CORRECTNESS BY CONDITION AND PROMPT")
print("=" * 78)
piv = df.pivot_table(index="cond", columns="prompt", values="correct", aggfunc="mean")
piv["mean"] = piv.mean(axis=1)
piv["spread"] = piv[list(PROMPTS)].max(axis=1) - piv[list(PROMPTS)].min(axis=1)
piv = piv.reindex([c for c in COND if c in piv.index])
piv.insert(0, "condition", [COND_LABEL[c] for c in piv.index])
print(piv.to_string(float_format=lambda x: f"{x:.3f}"))
piv.to_csv(os.path.join(OUT, "prompt_sensitivity_summary.csv"))

rng = np.random.default_rng(CONFIG["seed"])
def paired(a, b, pid, col="correct"):
    s = df[df.prompt == pid]
    x = s[s.cond == a].set_index("qid")[col]; y = s[s.cond == b].set_index("qid")[col]
    i = x.index.intersection(y.index)
    d = (x.loc[i] - y.loc[i]).values.astype(float)
    m = d[rng.integers(0, len(d), size=(CONFIG["bootstrap_n"], len(d)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return d.mean(), lo, hi

print("\n" + "=" * 78)
print("DOES THE CONCLUSION SURVIVE EVERY PROMPT?")
print("=" * 78)
rows = []
pairs = [("B", "A", "dialect gap")] + ([("F", "B", "reranked k=1 vs k=5")] if "F" in CONFIG["conditions"] else [])
for a, b, lab in pairs:
    if a not in set(df.cond) or b not in set(df.cond):
        continue
    for pid in PROMPTS:
        d, lo, hi = paired(a, b, pid)
        sig = "yes" if (lo > 0 or hi < 0) else "no"
        rows.append({"comparison": lab, "prompt": pid, "diff": d, "lo": lo, "hi": hi, "significant": sig})
S = pd.DataFrame(rows)
print(S.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
S.to_csv(os.path.join(OUT, "prompt_sensitivity_comparisons.csv"), index=False)

for lab, g in S.groupby("comparison"):
    same_sign = (g["diff"] > 0).all() or (g["diff"] < 0).all()
    same_sig = g.significant.nunique() == 1
    print(f"\n  {lab}: sign consistent across prompts: {'yes' if same_sign else 'NO'}; "
          f"significance consistent: {'yes' if same_sig else 'NO'}")
    if not (same_sign and same_sig):
        print("    The conclusion is prompt-dependent and must be reported as a range, "
              "not as a single value.")

lines = []
for c in piv.index:
    vals = " & ".join(f"{piv.loc[c, p]:.3f}" for p in PROMPTS)
    lines.append(f"{COND_LABEL[c].replace('·','&')} & {vals} & {piv.loc[c,'spread']:.3f}\\\\")
open("tables/table7_prompt_sensitivity.tex", "w").write("\n".join(lines) + "\n")
print("\nsaved tables/table7_prompt_sensitivity.tex")
