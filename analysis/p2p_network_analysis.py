import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ─── Load ─────────────────────────────────────────────────────────────────────
CSV_PATH = "/home/jovyan/multi_agent_benchmark/benchmarks/results/p2p_results_benchmark.csv"
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["total_time_ms"]).copy()
df["total_time_ms"]   = pd.to_numeric(df["total_time_ms"],   errors="coerce")
df["total_exchanges"] = pd.to_numeric(df["total_exchanges"], errors="coerce")
df.dropna(subset=["total_time_ms"], inplace=True)

CONDITIONS = [c for c in [
    "baseline", "100ms_delay", "jitter_50ms_gaussian",
    "packet_loss_5percent", "iot_edge_extreme"
] if c in df["network_condition"].unique()]

COLORS = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c"]
CMAP   = dict(zip(CONDITIONS, COLORS))

# ─── Peer pair label ──────────────────────────────────────────────────────────
df["pair"] = df["my_id"] + " ↔ " + df["peer_id"]

# ─── Anomaly Detection (per condition per pair) ───────────────────────────────
df["is_anomaly"] = False
for cond in CONDITIONS:
    idx = df[df["network_condition"] == cond].index
    if len(idx) < 10:
        continue
    s      = df.loc[idx, "total_time_ms"]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr    = q3 - q1
    z      = np.abs(stats.zscore(s))
    df.loc[idx, "is_anomaly"] = (
        (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr) | (z > 3)
    )
# ─── Print Summary ────────────────────────────────────────────────────────────
print("=" * 55)
print("P2P PATTERN — NETWORK ANALYSIS")
print("=" * 55)
for cond in CONDITIONS:
    s    = df[df["network_condition"] == cond]["total_time_ms"]
    anom = df[(df["network_condition"] == cond) & df["is_anomaly"]].shape[0]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr    = q3 - q1
    print(f"\n  Condition : {cond}")
    print(f"  Mean Time : {s.mean():.2f} ms  |  Std: {s.std():.2f} ms")
    print(f"  Median    : {s.median():.2f} ms  |  P95: {np.percentile(s, 95):.2f} ms")
    print(f"  IQR range : [{q1 - 1.5*iqr:.2f}, {q3 + 1.5*iqr:.2f}] ms")
    print(f"  Anomalies : {anom} / {len(s)} ({anom/len(s)*100:.1f}%)")

# ─── Plots ────────────────────────────────────────────────────────────────────
cols = [CMAP[c] for c in CONDITIONS]
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("P2P Pattern — Network Analysis & Anomaly Detection",
             fontsize=14, fontweight="bold")

# ── 1. Mean total_time bar ────────────────────────────────────────────────────
ax = axes[0, 0]
means = [df[df["network_condition"] == c]["total_time_ms"].mean() for c in CONDITIONS]
stds  = [df[df["network_condition"] == c]["total_time_ms"].std()  for c in CONDITIONS]
ax.bar(CONDITIONS, means, color=cols, alpha=0.85, yerr=stds, capsize=4)
ax.set_title("Mean Total Time by Condition")
ax.set_ylabel("ms")
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(bottom=0)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 5, f"{m:.0f}", ha="center", fontsize=8)

# ── 2. Box plot ───────────────────────────────────────────────────────────────
ax = axes[0, 1]
box_data = [df[df["network_condition"] == c]["total_time_ms"].values for c in CONDITIONS]
bp = ax.boxplot(box_data, patch_artist=True,
                medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], cols):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_xticklabels(CONDITIONS, rotation=20, fontsize=8)
ax.set_title("Total Time Distribution")
ax.set_ylabel("ms")
ax.grid(axis="y", alpha=0.3)

# ── 3. Total time by peer pair × condition ────────────────────────────────────
ax     = axes[0, 2]
pairs  = df["pair"].unique()
x      = np.arange(len(CONDITIONS))
width  = 0.8 / len(pairs)
pcolors = ["#1abc9c", "#e74c3c", "#f39c12", "#8e44ad"]
for i, pair in enumerate(pairs):
    means_p = [
        df[(df["network_condition"] == c) & (df["pair"] == pair)]["total_time_ms"].mean()
        for c in CONDITIONS
    ]
    ax.bar(x + i * width, means_p, width, label=pair,
           color=pcolors[i % len(pcolors)], alpha=0.85)
ax.set_xticks(x + width * (len(pairs) - 1) / 2)
ax.set_xticklabels(CONDITIONS, rotation=20, fontsize=8)
ax.set_title("Total Time by Peer Pair × Condition")
ax.set_ylabel("ms")
ax.legend(fontsize=7)
ax.grid(axis="y", alpha=0.3)

# ── 4. Total time by prompt type × condition ──────────────────────────────────
ax     = axes[1, 0]
ptypes = df["prompt_type"].unique()
width  = 0.8 / len(ptypes)
tcolors = ["#3498db", "#e74c3c", "#2ecc71"]
for i, pt in enumerate(ptypes):
    means_t = [
        df[(df["network_condition"] == c) & (df["prompt_type"] == pt)]["total_time_ms"].mean()
        for c in CONDITIONS
    ]
    ax.bar(x + i * width, means_t, width, label=pt,
           color=tcolors[i % len(tcolors)], alpha=0.85)
ax.set_xticks(x + width * (len(ptypes) - 1) / 2)
ax.set_xticklabels(CONDITIONS, rotation=20, fontsize=8)
ax.set_title("Total Time by Prompt Type × Condition")
ax.set_ylabel("ms")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# ── 5. Avg exchanges per condition ────────────────────────────────────────────
ax = axes[1, 1]
avg_exchanges = [
    df[df["network_condition"] == c]["total_exchanges"].mean()
    for c in CONDITIONS
]
ax.bar(CONDITIONS, avg_exchanges, color=cols, alpha=0.85)
ax.set_title("Avg Exchanges per Condition")
ax.set_ylabel("Exchanges")
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(bottom=0)
for i, v in enumerate(avg_exchanges):
    ax.text(i, v + 0.02, f"{v:.1f}", ha="center", fontsize=8)

# ── 6. Anomaly scatter ────────────────────────────────────────────────────────
ax = axes[1, 2]
for cond, color in zip(CONDITIONS, cols):
    normal = df[(df["network_condition"] == cond) & (~df["is_anomaly"])]
    anom   = df[(df["network_condition"] == cond) &  (df["is_anomaly"])]
    ax.scatter([cond] * len(normal), normal["total_time_ms"],
               color=color, alpha=0.4, s=25)
    ax.scatter([cond] * len(anom), anom["total_time_ms"],
               color="red", marker="x", s=90, linewidths=2)
ax.set_title("Anomalies (Red ✕)")
ax.set_ylabel("Total Time (ms)")
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("/home/jovyan/multi_agent_benchmark/analysis/p2p_analysis.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved → analysis/p2p_analysis.png")