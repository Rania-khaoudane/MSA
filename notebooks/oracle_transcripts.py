# =============================================================================
# Minor point: explain why the oracle condition underperforms MSA retrieval.
#
# check_oracle_anomaly.py established that the oracle (H) is significantly
# below A (MSA retrieval) on correctness, but not below B, D or E. That needs
# an explanation in the Discussion, and the explanation depends on WHY the
# oracle's answers were scored wrong.
#
# This pulls every oracle item scored incorrect into a readable file and
# pre-classifies each one, so the manual read is a check of the classification
# rather than a cold start. The categories distinguish the two explanations
# that lead to very different paper text:
#   - judge artefact (correct answer, penalised for wording) -> a judge-quality
#     issue, tied to point A
#   - genuine failure (wrong or absent answer despite a correct passage) -> a
#     real finding about short single-passage contexts
#
# No GPU needed.
# =============================================================================
import json, re
import numpy as np
import pandas as pd

CONFIG = {"gen_raw": "rag_outputs/rag_generation_raw.csv",
          "qa_file": "qa_pairs_wiki.json",
          "out": "oracle_transcripts.csv",
          "compare_cond": "A"}

qa = {q["id"]: q for q in json.load(open(CONFIG["qa_file"], encoding="utf-8"))}
df = pd.read_csv(CONFIG["gen_raw"])

DIAC = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED\u0670]")
def norm(t):
    t = DIAC.sub("", str(t or ""))
    for a, b in [(r"[\u0625\u0623\u0622\u0627]", "\u0627"), (r"\u0649", "\u064A"),
                 (r"\u0629", "\u0647"), (r"\u0640+", ""), (r"[^\w\s]", " ")]:
        t = re.sub(a, b, t)
    return re.sub(r"\s+", " ", t).strip()
def nums(t): return set(re.findall(r"\d[\d.,٬]*", str(t or "")))

def classify(gold, ans):
    """A first pass at why this was scored wrong, to be confirmed by reading."""
    if "غير متوفرة" in str(ans):
        return "refusal (model declined despite having the gold passage)"
    g, a = set(norm(gold).split()), set(norm(ans).split())
    gn, an = nums(gold), nums(ans)
    if gn and an and gn & an:
        return "LIKELY JUDGE ARTEFACT: correct number present, wording differs"
    if gn and an and not (gn & an):
        return "genuine error: different numbers"
    if g and len(g & a) / len(g) >= 0.5:
        return "LIKELY JUDGE ARTEFACT: high overlap with gold wording"
    if len(str(ans).strip()) < 10:
        return "genuine error: answer too short / empty"
    return "unclear -- read this one"

oracle = df[df.cond == "H"]
wrong = oracle[oracle.correct == 0].copy()
print(f"Oracle items scored incorrect: {len(wrong)} of {len(oracle)}")

other = df[df.cond == CONFIG["compare_cond"]].set_index("qid")
rows = []
for _, r in wrong.iterrows():
    q = qa.get(r.qid, {})
    gold = q.get("gold_answer", "")
    rows.append({
        "qid": r.qid,
        "question_msa": q.get("msa_query", ""),
        "gold_answer": gold,
        "oracle_answer": r.answer,
        "classification": classify(gold, r.answer),
        f"{CONFIG['compare_cond']}_answer": other.answer.get(r.qid, ""),
        f"{CONFIG['compare_cond']}_correct": other.correct.get(r.qid, np.nan),
    })
T = pd.DataFrame(rows)
T.to_csv(CONFIG["out"], index=False, encoding="utf-8-sig")

print("\nPre-classification (confirm by reading, do not take on trust):")
print(T.classification.value_counts().to_string())

artefact = T.classification.str.startswith("LIKELY JUDGE ARTEFACT").mean()
print(f"""
  {artefact:.0%} of the oracle's failures look like judge artefacts rather than
  genuine wrong answers.

  If that share is high, the oracle result is mostly a measurement problem and
  belongs with point A (judge validity), not as a claim about context length.
  If it is low, the generator genuinely struggles with a single short passage
  even when that passage is correct, which is a real finding for the Discussion.

  Wrote {CONFIG['out']} -- open it and read 10-15 rows before writing either claim.
""")

interesting = T[(T[f"{CONFIG['compare_cond']}_correct"] == 1)]
print(f"{len(interesting)} items where the oracle failed but "
      f"{CONFIG['compare_cond']} succeeded -- these are the ones driving the gap. First 5:\n")
for _, r in interesting.head(5).iterrows():
    print(f"  [{r.qid}] {r.classification}")
    print(f"    gold:   {str(r.gold_answer)[:90]}")
    print(f"    oracle: {str(r.oracle_answer)[:90]}")
    print(f"    {CONFIG['compare_cond']}:      {str(r[f'{CONFIG['compare_cond']}_answer'])[:90]}\n")
