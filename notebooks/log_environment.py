# =============================================================================
# Record exactly how the reported results were produced (reviewer points C, D).
#
# Writes run_environment.json: library versions, hardware, quantisation,
# decoding, the validated judge, and SHA-256 checksums of the input data and of
# every file in final_results/ -- the folder all reported numbers come from.
# Cite this file in the Methods and archive it with the results.
#
# RUN (in the pytorch environment, from the project folder):
#   python3 log_environment.py
# =============================================================================
import json, sys, platform, subprocess, hashlib, os
from importlib import metadata

def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None

def version(dist):
    # importlib.metadata finds versions even for packages without __version__
    # (rank_bm25 is one of them).
    try:
        return metadata.version(dist)
    except Exception:
        return None

def sha256(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def judge_kappa():
    path = "judge_improvement/judge_selection.csv"
    if not os.path.exists(path):
        return None
    try:
        import pandas as pd
        js = pd.read_csv(path)
        return {r.config: {"kappa": round(float(r.kappa), 3),
                           "agreement": round(float(r.agreement), 3),
                           "parse_rate": round(float(r.parse_rate), 3)}
                for _, r in js.iterrows()}
    except Exception:
        return None

final_files = sorted(os.path.join(r, n) for r, _, fs in os.walk("final_results")
                     for n in fs) if os.path.isdir("final_results") else []

env = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "packages": {d: version(d) for d in
                 ["torch", "transformers", "sentence-transformers", "accelerate",
                  "bitsandbytes", "numpy", "pandas", "rank-bm25", "scikit-learn"]},
    "gpu": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
    "models": {
        "retrieval_encoder": "intfloat/multilingual-e5-base",
        "hybrid_weight_alpha": 0.8,
        "reranker": "BAAI/bge-reranker-v2-m3 (top-20 candidates)",
        "generator": "Qwen/Qwen2.5-7B-Instruct",
        "second_generator": "MBZUAI-Paris/Atlas-Chat-9B",
    },
    "judge": {
        "model": "MBZUAI-Paris/Atlas-Chat-9B",
        "prompt": "v2 (explicit refusal and paraphrase rules)",
        "selection": "chosen among four configurations on 96 human-labelled answers",
        "kappa_by_configuration": judge_kappa(),
        "note": "the second generator's answers are evaluated with token-F1 "
                "(judge-free), because Atlas-Chat-9B judging its own answers "
                "showed self-preference",
    },
    "quantisation": {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "note": "compute dtype is pinned: float32 for the non-quantised modules "
                "changes most generated answers and shifts correctness materially",
    },
    "decoding": {"do_sample": False, "max_new_tokens_generation": 96,
                 "max_new_tokens_judge": 32,
                 "note": "greedy decoding; temperature/top_p/top_k are unset and ignored"},
    "statistics": {"bootstrap_resamples": 4000, "seed": 42,
                   "multiple_comparisons": "Holm-Bonferroni within each metric",
                   "primary_hypothesis": "MSA vs Darija Recall@1 (uncorrected)"},
    "data_checksums": {f: sha256(f) for f in ["corpus_v2.json", "qa_pairs_wiki.json"]},
    "final_results_checksums": {f: sha256(f) for f in final_files},
}
try:
    import torch
    env["cuda_runtime"] = torch.version.cuda
    env["torch_cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
except Exception:
    pass

json.dump(env, open("run_environment.json", "w"), indent=2)
print(json.dumps({k: env[k] for k in ["packages", "judge", "models"]}, indent=2))
print(f"\nchecksummed {len(final_files)} files in final_results/")
missing = [d for d, v in env["packages"].items() if v is None]
if missing:
    print(f"WARNING: no version found for {missing} -- are you in the pytorch environment?")
print("wrote run_environment.json -- cite it in the Methods and archive it with the results.")
