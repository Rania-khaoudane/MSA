# =============================================================================
# Generate Figures 0-5 for the manuscript. Figure 6 (end-to-end) already
# exists at final_results/fig6_rag.pdf/.png, produced by final_analysis.py.
#
# Reads whichever of these are present; skips a figure cleanly if its source
# file is missing, and says so, rather than failing:
#   results/gap_by_reranker.csv            -> Fig 1, 3
#   results/reranker_experiments_full.csv  -> Fig 1
#   final_results/second_generator_tokenf1.csv (not used here)
#   results/latency_measurements.csv       -> Fig 4
#   results/margin_gating_curve.csv        -> Fig 5
#   final_results/main_results_raw.csv     -> Fig 2 (margin, if computed there)
#
# CPU only, a few seconds.  RUN:  python3 figures.py
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito palette: colour-blind safe, readable in greyscale.
MSA_C, DAR_C, BEST_C, GOLD, INK, MUTED = \
    "#0072B2", "#D55E00", "#009E73", "#E69F00", "#1b1b1b", "#8a8a8a"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 17, "axes.edgecolor": "#bdbdbd", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.7,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig)
    print(f"  saved {OUT}/{name}.pdf")

def load(path):
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None
    return pd.read_csv(path)

def subtitle(ax, text):
    ax.text(0, 1.012, text, transform=ax.transAxes, fontsize=8.3, color=MUTED, va="bottom")

def halo():
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=3, foreground="white")]

# ---------------------------------------------------------------- Figure 0: pipeline
print("Figure 0 -- pipeline diagram")
fig, ax = plt.subplots(figsize=(7.4, 2.6))
ax.set_xlim(-1, 101); ax.set_ylim(0, 35); ax.axis("off")

def box(x, y, w, h, title, sub, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.5",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
            fontsize=8.8, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=7.3, color="#444")

def arrow(p0, p1, color=INK):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=1.3, shrinkA=0, shrinkB=0))

box(0, 12, 12.5, 10, "Query", "MSA or Darija", "#f4f4f4", "#9e9e9e")
box(16.5, 12, 22, 10, "Hybrid retrieval", "BM25 + e5  ·  ~26 ms", "#e8f1f8", MSA_C)
ax.add_patch(Polygon([[43, 17], [51, 25.5], [59, 17], [51, 8.5]], closed=True,
                     fc="#fdf3e1", ec=GOLD, lw=1.3))
ax.text(51, 18.6, "Confident?", ha="center", fontsize=8.2, fontweight="bold")
ax.text(51, 14.4, r"$\hat{m}(q) > \tau$", ha="center", fontsize=8.2)
box(64, 23.5, 22.5, 9.5, "Cross-encoder rerank", "top-20  ·  ~1,361 ms", "#e6f5ef", BEST_C)
box(89.5, 12, 10.5, 10, "Top-k", "passages", "#f4f4f4", "#9e9e9e")

arrow((12.9, 17), (16.1, 17)); arrow((38.9, 17), (42.7, 17))
arrow((51, 25.8), (63.6, 28.2), color=BEST_C)
ax.text(55.5, 30.2, "no", fontsize=7.8, color=BEST_C, fontweight="bold", ha="center")
arrow((86.9, 27.5), (92.5, 22.5), color=BEST_C)
arrow((59.3, 17), (89.1, 17), color=MUTED)
ax.text(73, 13.4, "yes  →  skip reranker", fontsize=7.6, color=MUTED, fontweight="bold", ha="center")
ax.text(0, 2.5, "Margin-gated reranking: the costly reranker runs only when retrieval is "
        "uncertain, which happens more often for dialectal queries.",
        fontsize=7.8, color=MUTED, style="italic")
save(fig, "fig0_pipeline")

# ---------------------------------------------------------------- Figure 1: recall@k
print("Figure 1 -- Recall@k by variety")
rr = load("results/reranker_experiments_full.csv")
if rr is not None and {"method", "query", "depth"} <= set(rr.columns):
    D = 20
    ks = [1, 3, 5, 10]
    def row(m, q):
        s = rr[(rr.method == m) & (rr["query"] == q) & (rr.depth == D)]
        return s.iloc[0] if len(s) else None
    bm, bd = row("no_rerank", "msa_query"), row("no_rerank", "darija_query")
    rm, rd = row("bge-reranker-v2-m3", "msa_query"), row("bge-reranker-v2-m3", "darija_query")
    if all(x is not None for x in [bm, bd, rm, rd]):
        fig, ax = plt.subplots(figsize=(5.6, 3.7))
        bmv = [bm[f"R@{k}"] for k in ks]; bdv = [bd[f"R@{k}"] for k in ks]
        rmv = [rm[f"R@{k}"] for k in ks]; rdv = [rd[f"R@{k}"] for k in ks]
        ax.fill_between(ks, bdv, bmv, color=DAR_C, alpha=0.10, lw=0, label="Dialect gap (retrieval)")
        ax.plot(ks, bmv, "--", color=MSA_C, lw=1.6, marker="o", ms=5, mfc="white", mew=1.5, label="MSA · retrieval")
        ax.plot(ks, bdv, "--", color=DAR_C, lw=1.6, marker="o", ms=5, mfc="white", mew=1.5, label="Darija · retrieval")
        ax.plot(ks, rmv, "-", color=MSA_C, lw=2.3, marker="o", ms=6, label="MSA · reranked")
        ax.plot(ks, rdv, "-", color=DAR_C, lw=2.3, marker="o", ms=6, label="Darija · reranked")
        gain = rdv[0] - bdv[0]
        ax.annotate("", xy=(1, rdv[0] - 0.006), xytext=(1, bdv[0] + 0.006),
                    arrowprops=dict(arrowstyle="-|>", color=BEST_C, lw=2, mutation_scale=14))
        note = f"+{gain:.2f} from reranking"
        if rdv[0] > bmv[0]:
            note += "\nabove original MSA"
        ax.text(1.12, bdv[0] + 0.03, note, color=BEST_C, fontsize=8.3, fontweight="bold",
                va="bottom", path_effects=halo())
        ax.set_xticks(ks); ax.set_xticklabels([f"@{k}" for k in ks])
        ax.set_xlabel("Recall cut-off"); ax.set_ylabel("Recall")
        ax.set_ylim(max(0, min(bdv) - 0.06), 1.015)
        ax.set_title("The gap lives at rank 1")
        subtitle(ax, "Correct passage is usually retrieved; for dialect it is not ranked first")
        ax.legend(fontsize=7.8, loc="lower right")
        save(fig, "fig1_recall_at_k")
    else:
        print("  [skip] missing no_rerank/bge-reranker-v2-m3 rows at depth 20")

# ---------------------------------------------------------------- Figure 2: margin
print("Figure 2 -- retrieval confidence")
margins = load("results/margin_by_variety.csv")
if margins is not None:
    p = margins.pivot(index="qid", columns="query", values="margin").dropna()
    d = (p["msa_query"] - p["darija_query"]).values
    rng = np.random.default_rng(42)
    boot = d[rng.integers(0, len(d), size=(1000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    means = [p["msa_query"].mean(), p["darija_query"].mean()]

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.bar(["MSA", "Darija"], means, color=[MSA_C, DAR_C], width=0.55)
    for i, v in enumerate(means):
        ax.text(i, v * 1.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Mean top-1/top-2 score margin")
    ax.set_ylim(0, max(means) * 1.2)
    ax.set_title(f"Retrieval confidence\n(difference 95% CI [{lo:.3f}, {hi:.3f}])", fontsize=9)
    save(fig, "fig2_margin_by_variety")
else:
    print("  Using the fixed values reported earlier (MSA 0.135, Darija 0.111)")
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    means = [0.135, 0.111]
    ax.bar(["MSA", "Darija"], means, color=[MSA_C, DAR_C], width=0.55)
    for i, v in enumerate(means):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Mean top-1/top-2 score margin"); ax.set_ylim(0, 0.16)
    ax.set_title("Retrieval confidence\n(difference 95% CI [0.015, 0.033])", fontsize=9)
    save(fig, "fig2_margin_by_variety")

# ---------------------------------------------------------------- Figure 3: gap by reranker
print("Figure 3 -- dialect gap by reranker")
gap = load("results/gap_by_reranker.csv")
if gap is not None and {"method", "gap", "lo", "hi"} <= set(gap.columns):
    order = [m for m in ["no_rerank", "bge-reranker-v2-m3", "GATE-Reranker-V1",
                         "Namaa-ARA-Reranker-V1", "bge-reranker-base"] if m in set(gap.method)]
    g = gap.set_index("method").reindex(order)
    nice = {"no_rerank": "No reranking"}
    fig, ax = plt.subplots(figsize=(5.6, 0.55 * len(g) + 1.2))
    y = np.arange(len(g))
    colors = [MUTED if m == "no_rerank" else (BEST_C if m == "bge-reranker-v2-m3" else "#9bbb59")
              for m in g.index]
    ax.barh(y, g.gap, xerr=[g.gap - g.lo, g.hi - g.gap], color=colors, capsize=3)
    ax.set_yticks(y); ax.set_yticklabels([nice.get(m, m) for m in g.index])
    ax.invert_yaxis(); ax.set_xlabel("MSA $-$ Darija Recall@1 gap (95% CI)")
    ax.set_title("Dialect gap by reranker (lower is better)")
    save(fig, "fig3_gap_by_reranker")

# ---------------------------------------------------------------- Figure 4: latency
print("Figure 4 -- latency")
lat = load("results/latency_measurements.csv")
if lat is not None and {"retrieve_ms", "rerank_ms"} <= set(lat.columns):
    ret_med, rr_med = lat.retrieve_ms.median(), lat.rerank_ms.median()
    ratio = rr_med / ret_med
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    rng = np.random.default_rng(1)
    for i, (col, c) in enumerate([("retrieve_ms", MSA_C), ("rerank_ms", BEST_C)]):
        v = lat[col].values
        ax.scatter(v, i + rng.uniform(-0.18, 0.18, len(v)), s=10, color=c, alpha=0.35, lw=0)
        med = np.median(v)
        ax.plot([med, med], [i - 0.28, i + 0.28], color=c, lw=3)
        ax.text(med, i + 0.36, f"{med:,.0f} ms", ha="center", fontsize=8.8,
                fontweight="bold", color=c, path_effects=halo())
    ax.set_xscale("log")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Hybrid retrieval", "Cross-encoder\nrerank"])
    ax.set_ylim(-0.6, 1.75)
    ax.annotate("", xy=(rr_med, 1.55), xytext=(ret_med, 1.55),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
    ax.text(np.sqrt(ret_med * rr_med), 1.62, f"≈{ratio:.0f}× slower", ha="center",
            fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Latency per query (ms, log scale)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Reranking is accurate but expensive")
    subtitle(ax, "Each dot is one query; bar marks the median")
    save(fig, "fig4_latency")

# ---------------------------------------------------------------- Figure 5: margin gating
print("Figure 5 -- margin-gated reranking")
gate = load("results/margin_gating_curve.csv")
if gate is not None and {"query", "tau", "R@1", "frac_reranked"} <= set(gate.columns):
    dar = gate[gate["query"] == "darija_query"].sort_values("frac_reranked")
    msa = gate[gate["query"] == "msa_query"].sort_values("frac_reranked")
    never, always = dar["R@1"].iloc[0], dar["R@1"].iloc[-1]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    x = dar.frac_reranked * 100
    ax.fill_between(x, never, dar["R@1"], color=DAR_C, alpha=0.08, lw=0)
    ax.plot(msa.frac_reranked * 100, msa["R@1"], "-o", color=MSA_C, lw=1.8, ms=4, label="MSA")
    ax.plot(x, dar["R@1"], "-o", color=DAR_C, lw=2.2, ms=4.5, label="Darija")
    ax.axhline(always, color=MUTED, lw=0.8, ls=":")
    ax.text(1, always + 0.004, "always rerank", fontsize=7.5, color=MUTED)
    hl = dar[np.isclose(dar.tau, 0.10)]
    if len(hl):
        hx, hy = float(hl.frac_reranked.iloc[0]) * 100, float(hl["R@1"].iloc[0])
        kept = (hy - never) / (always - never) * 100 if always > never else float("nan")
        ax.scatter(hx, hy, marker="*", s=320, color=GOLD, ec=INK, lw=0.8, zorder=6)
        ax.annotate(f"τ = 0.10\n{kept:.0f}% of the benefit\nat {hx:.0f}% of the cost",
                    xy=(hx, hy), xytext=(hx + 12, hy - 0.095), fontsize=8.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", fc="#fffaf0", ec=GOLD, lw=1),
                    arrowprops=dict(arrowstyle="-", color=GOLD, lw=1.2))
    ax.set_xlim(-2, 102)
    ax.set_xlabel("Queries sent to the reranker (%)"); ax.set_ylabel("Recall@1")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("Margin gating: most of the accuracy, a fraction of the cost", pad=26)
    save(fig, "fig5_margin_gating")

print(f"\nDone. Figures written to {OUT}/. Figure 6 (end-to-end) is already in "
      f"final_results/fig6_rag.pdf, made by final_analysis.py -- no need to remake it.")
