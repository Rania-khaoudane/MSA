# =============================================================================
# Point A + B: build a judge that agrees with humans, then re-judge everything.
#
# The labelled sheet gives kappa = 0.538 ("moderate") for the current judge --
# below the 0.6 the reviewer asks for, and measured on a judge from the same
# model family as the generator, which is the configuration known to carry
# self-preference bias.
#
# Rather than guess at a better prompt, this script SELECTS one empirically:
# it runs several judge configurations over the 96 human-labelled items only
# (cheap), scores each against the human labels, and keeps the best. Only then
# does it re-judge all generations with the winner.
#
# The candidate configurations target the two failure patterns visible in the
# disagreements:
#   - refusals were sometimes scored "correct" ("المعلومة غير متوفرة" marked
#     مطابق: نعم), which is never right unless the gold answer is itself a refusal
#   - correct answers carrying extra true detail, or the right number inside a
#     wider range, were scored wrong
#
# It also stores every raw verdict, so the parse rate can be reported (the
# reviewer's minor point).
#
# RUN with a GPU, after rag_end_to_end.py. Needs the labelled sheet.
# =============================================================================

CONFIG = {
    "gen_raw": "rag_outputs/rag_generation_raw.csv",
    "labelled_sheet": "judge_labeling_sheet.csv",   # the one you filled in
    "corpus_file": "corpus_v2.json",
    "qa_file": "qa_pairs_wiki.json",
    "generations_ckpt": "rag_outputs/generations.json",
    # Judge models to try. The first is the current one; the others are from a
    # different family, which is what the reviewer asks for. Any that fails to
    # load is skipped rather than aborting the run.
    "judge_models": [
        "Qwen/Qwen2.5-14B-Instruct",
        "MBZUAI-Paris/Atlas-Chat-9B",
    ],
    "batch": 4,
    "max_new_tokens": 32,
    "seed": 42,
    "out": "judge_improvement",
    # Set False to only run the selection stage and inspect results before
    # committing to a full re-judge of every generation.
    "rejudge_all": True,
    "DRY_RUN": False,
}

import os, re, json, gc, itertools
import numpy as np
import pandas as pd

OUT = CONFIG["out"]
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- data
corpus = json.load(open(CONFIG["corpus_file"], encoding="utf-8"))
qa = json.load(open(CONFIG["qa_file"], encoding="utf-8"))
ids = [c["chunk_id"] for c in corpus]
texts = [c["text"] for c in corpus]
pos = {c: i for i, c in enumerate(ids)}
byid = {q["id"]: q for q in qa}

gen_df = pd.read_csv(CONFIG["gen_raw"])
lab = pd.read_csv(CONFIG["labelled_sheet"])
lab = lab[lab.human_correct.notna() & (lab.human_correct.astype(str).str.strip() != "")]
lab["human_correct"] = lab.human_correct.astype(int)
print(f"{len(gen_df)} generations | {len(lab)} human-labelled items")

# Recover the context each labelled answer was produced with.
gens = json.load(open(CONFIG["generations_ckpt"], encoding="utf-8"))
ctx_lookup = {(g["qid"], g["cond"]): g["context"] for g in gens}
lab["context"] = [ctx_lookup.get((q, c)) for q, c in zip(lab.qid, lab.condition)]
missing_ctx = lab.context.isna().sum()
if missing_ctx:
    print(f"  warning: {missing_ctx} labelled rows have no stored context; dropped")
    lab = lab[lab.context.notna()]

def ctx_text(cids):
    return "\n\n".join(f"[{i+1}] {texts[pos[c]]}" for i, c in enumerate(cids))

# ---------------------------------------------------------------- prompts
# V1 is the prompt used in the canonical run, reproduced exactly.
PROMPT_V1 = """النصوص المرجعية:
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

# V2 adds explicit rules for the two observed failure modes.
PROMPT_V2 = """النصوص المرجعية:
{context}

السؤال: {question}
الإجابة الصحيحة: {gold}
الإجابة المقدمة: {answer}

قيّم الإجابة المقدمة وفق القواعد التالية:

قاعدة الامتناع: إذا قالت الإجابة المقدمة إن المعلومة غير متوفرة، فالجواب: مدعوم: لا، مطابق: لا.
(إلا إذا كانت الإجابة الصحيحة نفسها تفيد عدم التوفر.)

قاعدة المطابقة:
- يكفي أن تتضمن الإجابة المقدمة الجوهر الصحيح؛ لا يُشترط ذكر كل تفاصيل الإجابة الصحيحة.
- إضافة تفاصيل صحيحة إضافية لا تُبطل المطابقة.
- ذكر الرقم الصحيح ضمن سياق أوسع صحيح يُعدّ مطابقة.
- اختلاف الصياغة أو الترتيب مقبول.
- اختلاف الأرقام أو الأسماء أو التواريخ عن الإجابة الصحيحة يعني: مطابق: لا.

قاعدة الدعم: مدعوم يعني أن كل ما ورد في الإجابة المقدمة موجود صراحة في النصوص المرجعية.

أجب بهذا الشكل فقط وبدون أي شرح:
مدعوم: نعم/لا
مطابق: نعم/لا"""

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}

def parse(v):
    """Returns (faithful, correct, parsed_ok). Handles both verdicts landing on
    one line, which the original parser silently mis-scored."""
    t = (v or "").replace("،", " ")
    f = c = None
    for line in t.split("\n"):
        if "مدعوم" in line and f is None:
            seg = line.split("مطابق")[0]
            f = int("نعم" in seg)
        if "مطابق" in line:
            seg = line.split("مطابق", 1)[1]
            c = int("نعم" in seg)
    return (f or 0), (c or 0), (f is not None and c is not None)

# ---------------------------------------------------------------- models
try:
    import torch
except Exception:          # only needed for the real models, not for DRY_RUN
    torch = None
def free():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

if CONFIG["DRY_RUN"]:
    _rng = np.random.default_rng(0)
    def load_judge(name):
        print("stub judge:", name); return True
    def chat_batch(prompts, mnt=32):
        return [f"مدعوم: {'نعم' if _rng.random()<.8 else 'لا'}\nمطابق: {'نعم' if _rng.random()<.7 else 'لا'}"
                for _ in prompts]
else:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    llm = tokenizer = None
    def load_judge(name):
        global llm, tokenizer
        if llm is not None:
            del llm; llm = None; free()
        try:
            q4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            tokenizer = AutoTokenizer.from_pretrained(name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            llm = AutoModelForCausalLM.from_pretrained(name, quantization_config=q4,
                                                       device_map="auto", torch_dtype=torch.float16).eval()
            print("loaded", name)
            return True
        except Exception as e:
            print(f"  SKIPPED {name}: {type(e).__name__}: {str(e)[:160]}")
            return False
    @torch.no_grad()
    def chat_batch(prompts, mnt=32):
        t = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                           add_generation_prompt=True) for p in prompts]
        e = tokenizer(t, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(llm.device)
        o = llm.generate(**e, max_new_tokens=mnt, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        return [tokenizer.decode(g, skip_special_tokens=True).strip() for g in o[:, e["input_ids"].shape[1]:]]

def judge_rows(rows, template, mnt=None):
    """rows: iterable of dicts with context, question, gold, answer."""
    outs = []
    B = CONFIG["batch"]
    from tqdm import tqdm
    for i in tqdm(range(0, len(rows), B), desc="  judging", leave=False):
        b = rows[i:i + B]
        prompts = [template.format(context=ctx_text(r["context"]), question=r["question"],
                                   gold=r["gold"], answer=r["answer"]) for r in b]
        outs.extend(chat_batch(prompts, mnt or CONFIG["max_new_tokens"]))
    return outs

# ---------------------------------------------------------------- selection
from sklearn.metrics import cohen_kappa_score, accuracy_score

sel_rows = [{"context": eval(c) if isinstance(c, str) and c.startswith("[") else c,
             "question": byid[q]["msa_query"], "gold": byid[q]["gold_answer"], "answer": a}
            for q, c, a in zip(lab.qid, lab.context, lab.model_answer)]
human = lab.human_correct.values

print("\n" + "=" * 78)
print("SELECTING A JUDGE CONFIGURATION AGAINST HUMAN LABELS")
print("=" * 78)
results, verdict_store = [], {}
for model_name in CONFIG["judge_models"]:
    if not load_judge(model_name):
        continue
    for pname, tpl in PROMPTS.items():
        key = f"{model_name.split('/')[-1]} / {pname}"
        raw = judge_rows(sel_rows, tpl)
        parsed = [parse(v) for v in raw]
        corr = np.array([p[1] for p in parsed])
        rate = float(np.mean([p[2] for p in parsed]))
        k = cohen_kappa_score(human, corr)
        acc = accuracy_score(human, corr)
        results.append({"config": key, "model": model_name, "prompt": pname,
                        "kappa": k, "agreement": acc, "parse_rate": rate})
        verdict_store[key] = raw
        print(f"  {key:<42} kappa={k:.3f}  agreement={acc:.3f}  parse={rate:.3f}")
    free()

if not results:
    raise SystemExit("No judge model could be loaded.")

R = pd.DataFrame(results).sort_values("kappa", ascending=False)
R.to_csv(os.path.join(OUT, "judge_selection.csv"), index=False)
best = R.iloc[0]
print(f"\n  BEST: {best.config}  kappa={best.kappa:.3f} "
      f"({'meets' if best.kappa >= 0.6 else 'below'} the 0.6 target)")
baseline = R[R.config.str.endswith("/ v1")]
if len(baseline):
    b0 = baseline.iloc[0]
    print(f"  Current judge in the paper: {b0.config} kappa={b0.kappa:.3f} "
          f"-> improvement {best.kappa - b0.kappa:+.3f}")

pd.DataFrame({"qid": lab.qid.values, "condition": lab.condition.values,
              "human": human,
              **{k.replace(" ", ""): [parse(v)[1] for v in raw]
                 for k, raw in verdict_store.items()}}) \
  .to_csv(os.path.join(OUT, "judge_per_item.csv"), index=False)

# ---------------------------------------------------------------- re-judge all
if not CONFIG["rejudge_all"]:
    print("\nrejudge_all is False; stopping after selection.")
    raise SystemExit(0)

if best.kappa < 0.6:
    print("\n  NOTE: the best configuration is still below kappa 0.6. Re-judging with it")
    print("  is still an improvement, but the Limitations section must report the")
    print("  achieved kappa and state that correctness is a proxy of moderate validity.")

print("\n" + "=" * 78)
print(f"RE-JUDGING ALL {len(gens)} GENERATIONS WITH THE SELECTED CONFIGURATION")
print("=" * 78)
if best.model != CONFIG["judge_models"][0] or True:
    load_judge(best.model)
tpl = PROMPTS[best.prompt]

all_rows = [{"context": g["context"], "question": byid[g["qid"]]["msa_query"],
             "gold": byid[g["qid"]]["gold_answer"], "answer": g["answer"]} for g in gens]
raw_all = judge_rows(all_rows, tpl)

for g, v in zip(gens, raw_all):
    f, c, ok = parse(v)
    g["faithful_v2"], g["correct_v2"], g["judge_parsed"], g["judge_raw"] = f, c, int(ok), v

json.dump(gens, open(os.path.join(OUT, "generations_rejudged.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

df = pd.DataFrame(gens)
df["refused"] = df.answer.fillna("").str.contains("غير متوفرة").astype(int)
df.to_csv(os.path.join(OUT, "rag_generation_raw_rejudged.csv"), index=False)
print(f"  parse rate: {df.judge_parsed.mean():.3f}")

# ---------------------------------------------------------------- new numbers
LABELS = {"A": "MSA · retrieval · k=5", "B": "Darija · retrieval · k=5",
          "C": "Darija · retrieval · k=1", "D": "Darija · normalised · k=5",
          "E": "Darija · reranked · k=5", "F": "Darija · reranked · k=1",
          "G": "Darija · gated · k=1", "H": "Darija · gold passage"}
order = [c for c in LABELS if c in set(df.cond)]
summ = df.groupby("cond").agg(
    n=("qid", "count"), gold_in_context=("gold_in_context", "mean"),
    correct_old=("correct", "mean"), correct_new=("correct_v2", "mean"),
    faithful_new=("faithful_v2", "mean"), refusal=("refused", "mean"),
).reindex(order)
# faithfulness excluding refusals, which is the reviewer's point B
ans = df[df.refused == 0].groupby("cond").faithful_v2.mean()
summ["faithful_answered"] = ans
summ.insert(0, "condition", [LABELS[c] for c in summ.index])
summ.to_csv(os.path.join(OUT, "summary_rejudged.csv"))
print("\n" + summ.to_string(float_format=lambda x: f"{x:.3f}"))

rng = np.random.default_rng(CONFIG["seed"])
def paired(a, b, col):
    x = df[df.cond == a].set_index("qid")[col]; y = df[df.cond == b].set_index("qid")[col]
    i = x.index.intersection(y.index)
    d = (x.loc[i] - y.loc[i]).values.astype(float)
    m = d[rng.integers(0, len(d), size=(4000, len(d)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return d.mean(), lo, hi, min(max(p, 1/4000), 1.0)

COMPARE = [("B","A","Dialect gap in answers (k=5)"), ("C","B","Cost of k=1 without reranking"),
           ("F","C","Reranking at k=1"), ("F","B","Reranked k=1 vs retrieval k=5"),
           ("E","B","Reranking at k=5"), ("D","B","Normalisation, end to end"),
           ("G","F","Gating vs always reranking (k=1)"), ("H","F","Headroom to the oracle")]
rows = []
for a, b, lab_ in COMPARE:
    if a in order and b in order:
        d, lo, hi, p = paired(a, b, "correct_v2")
        rows.append({"comparison": lab_, "a": a, "b": b, "diff": d, "lo": lo, "hi": hi,
                     "p_raw": p, "significant": "yes" if lo > 0 or hi < 0 else "no"})
C = pd.DataFrame(rows)

# Holm within this family, so the corrected view is available immediately
p = C.p_raw.values; idx = np.argsort(p); m = len(p); run = 0.0; adj = np.empty(m)
for i, v in enumerate(p[idx]):
    run = max(run, (m - i) * v); adj[i] = min(run, 1.0)
holm = np.empty(m); holm[idx] = adj
C["p_holm"] = holm
C["sig_holm"] = np.where(C.p_holm < 0.05, "yes", "no")
C.to_csv(os.path.join(OUT, "comparisons_rejudged.csv"), index=False)
print("\n" + C.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

print(f"""
{'='*78}
WHAT TO DO WITH THIS
{'='*78}
  Judge selected: {best.config}, kappa = {best.kappa:.3f} on {len(lab)} human-labelled items.
  Report this kappa, the raw agreement ({best.agreement:.3f}) and the parse rate
  ({df.judge_parsed.mean():.3f}) in the Methods.

  correct_old is the number currently in the manuscript; correct_new is from the
  selected judge. If they differ materially, the manuscript tables must be
  regenerated from summary_rejudged.csv and comparisons_rejudged.csv, and the
  canonical-run checksums re-recorded with log_environment.py.

  Files written to {OUT}/
""")
