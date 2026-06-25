import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ─── Load ─────────────────────────────────────────────────────────────────────
CSV_PATH = "/home/jovyan/multi_agent_benchmark/benchmarks/results/coordinator_benchmark.csv"
df = pd.read_csv(CSV_PATH)
df = df[df["status"] == "success"].copy()
df["rtt_ms"]      = pd.to_numeric(df["rtt_ms"],      errors="coerce")
df["llm_time_ms"] = pd.to_numeric(df["llm_time_ms"], errors="coerce")
df.dropna(subset=["rtt_ms"], inplace=True)

CONDITIONS = [c for c in [
    "baseline", "100ms_delay", "jitter_50ms_gaussian",
    "packet_loss_5percent", "iot_edge_extreme"
] if c in df["network_condition"].unique()]

COLORS = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c"]
CMAP   = dict(zip(CONDITIONS, COLORS))

# ─── Anomaly Detection ────────────────────────────────────────────────────────
df["is_anomaly"] = False
for cond in CONDITIONS:
    for model in df["model"].unique():
        idx = df[(df["network_condition"] == cond) & (df["model"] == model)].index
        if len(idx) < 5:
            continue
        s      = df.loc[idx, "rtt_ms"]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr    = q3 - q1
        z      = np.abs(stats.zscore(s))
        df.loc[idx, "is_anomaly"] = (
            (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr) | (z > 3)
        )

# ─── Print Summary ────────────────────────────────────────────────────────────
print("=" * 55)
print("COORDINATOR PATTERN — NETWORK ANALYSIS")
print("=" * 55)
for cond in CONDITIONS:
    s = df[df["network_condition"] == cond]["rtt_ms"]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr    = q3 - q1
    anom   = df[(df["network_condition"] == cond) & df["is_anomaly"]].shape[0]
    print(f"\n  Condition : {cond}")
    print(f"  Mean RTT  : {s.mean():.2f} ms  |  Std: {s.std():.2f} ms")
    print(f"  Median    : {s.median():.2f} ms  |  P95: {np.percentile(s, 95):.2f} ms")
    print(f"  IQR range : [{q1 - 1.5*iqr:.2f}, {q3 + 1.5*iqr:.2f}] ms")
    print(f"  Anomalies : {anom} / {len(s)} ({anom/len(s)*100:.1f}%)")

# ─── Plots ────────────────────────────────────────────────────────────────────
cols = [CMAP[c] for c in CONDITIONS]
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Coordinator Pattern — Network Analysis & Anomaly Detection",
             fontsize=14, fontweight="bold")

# ── 1. Mean RTT bar ───────────────────────────────────────────────────────────
ax = axes[0, 0]
means = [df[df["network_condition"] == c]["rtt_ms"].mean() for c in CONDITIONS]
stds  = [df[df["network_condition"] == c]["rtt_ms"].std()  for c in CONDITIONS]
ax.bar(CONDITIONS, means, color=cols, alpha=0.85, yerr=stds, capsize=4)
ax.set_title("Mean RTT by Condition")
ax.set_ylabel("ms")
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.3)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 5, f"{m:.0f}", ha="center", fontsize=8)
ax.set_ylim(bottom=0)   # <-- ADD THIS LINE HERE

# ── 2. Box plot ───────────────────────────────────────────────────────────────
ax = axes[0, 1]
box_data = [df[df["network_condition"] == c]["rtt_ms"].values for c in CONDITIONS]
bp = ax.boxplot(box_data, patch_artist=True,
                medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], cols):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_xticklabels(CONDITIONS, rotation=20, fontsize=8)
ax.set_title("RTT Distribution")
ax.set_ylabel("ms")
ax.grid(axis="y", alpha=0.3)

# ── 3. RTT by model ───────────────────────────────────────────────────────────
ax     = axes[0, 2]
models = df["model"].unique()
x      = np.arange(len(CONDITIONS))
width  = 0.8 / len(models)
mcolors = ["#1abc9c", "#e74c3c", "#f39c12"]
for i, model in enumerate(models):
    means_m = [df[(df["network_condition"] == c) & (df["model"] == model)]["rtt_ms"].mean()
               for c in CONDITIONS]
    ax.bar(x + i * width, means_m, width, label=model,
           color=mcolors[i % len(mcolors)], alpha=0.85)
ax.set_xticks(x + width * (len(models) - 1) / 2)
ax.set_xticklabels(CONDITIONS, rotation=20, fontsize=8)
ax.set_title("RTT by Model × Condition")
ax.set_ylabel("ms")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# ── 4. RTT by prompt type ─────────────────────────────────────────────────────
ax     = axes[1, 0]
ptypes = df["prompt_type"].unique()
width  = 0.8 / len(ptypes)
pcolors = ["#3498db", "#e74c3c", "#2ecc71"]
for i, pt in enumerate(ptypes):
    means_p = [df[(df["network_condition"] == c) & (df["prompt_type"] == pt)]["rtt_ms"].mean()
               for c in CONDITIONS]
    ax.bar(x + i * width, means_p, width, label=pt,
           color=pcolors[i % len(pcolors)], alpha=0.85)
ax.set_xticks(x + width * (len(ptypes) - 1) / 2)
ax.set_xticklabels(CONDITIONS, rotation=20, fontsize=8)
ax.set_title("RTT by Prompt Type × Condition")
ax.set_ylabel("ms")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# ── 5. LLM time vs network overhead ──────────────────────────────────────────
ax = axes[1, 1]
mean_llm = [df[df["network_condition"] == c]["llm_time_ms"].mean() for c in CONDITIONS]
mean_rtt = [df[df["network_condition"] == c]["rtt_ms"].mean()      for c in CONDITIONS]
overhead = [r - l for r, l in zip(mean_rtt, mean_llm)]
ax.bar(CONDITIONS, mean_llm, color="#3498db", alpha=0.85, label="LLM Time")
ax.bar(CONDITIONS, overhead, bottom=mean_llm, color="#e74c3c", alpha=0.85, label="Net Overhead")
ax.set_title("LLM Time vs Network Overhead")
ax.set_ylabel("ms")
ax.tick_params(axis="x", rotation=20)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# ── 6. Anomaly scatter ────────────────────────────────────────────────────────
ax = axes[1, 2]
for cond, color in zip(CONDITIONS, cols):
    normal = df[(df["network_condition"] == cond) & (~df["is_anomaly"])]
    anom   = df[(df["network_condition"] == cond) &  (df["is_anomaly"])]
    ax.scatter([cond] * len(normal), normal["rtt_ms"],
               color=color, alpha=0.4, s=25)
    ax.scatter([cond] * len(anom), anom["rtt_ms"],
               color="red", marker="x", s=90, linewidths=2)
ax.set_title("Anomalies (Red ✕)")
ax.set_ylabel("RTT (ms)")
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("/home/jovyan/multi_agent_benchmark/analysis/coordinator_analysis.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved → analysis/coordinator_analysis.png")