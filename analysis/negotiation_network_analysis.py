import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ─── Load ─────────────────────────────────────────────────────────────────────
CSV_PATH = "/home/jovyan/multi_agent_benchmark/benchmarks/results/benchmark_negotiation_results.csv"
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["total_time_ms"]).copy()
df["total_time_ms"] = pd.to_numeric(df["total_time_ms"], errors="coerce")
df["rounds"]        = pd.to_numeric(df["rounds"],        errors="coerce")
df.dropna(subset=["total_time_ms"], inplace=True)

CONDITIONS = [c for c in [
    "baseline", "100ms_delay", "jitter_50ms_gaussian",
    "packet_loss_5percent", "iot_edge_extreme"
] if c in df["network_condition"].unique()]

COLORS = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c"]
CMAP   = dict(zip(CONDITIONS, COLORS))
cols   = [CMAP[c] for c in CONDITIONS]

# ─── Anomaly Detection (per condition) ───────────────────────────────────────
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
print("NEGOTIATION PATTERN — NETWORK ANALYSIS")
print("=" * 55)
for cond in CONDITIONS:
    sub  = df[df["network_condition"] == cond]
    s    = sub["total_time_ms"]
    anom = sub[sub["is_anomaly"]].shape[0]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr    = q3 - q1
    agreed_pct = sub["agreed"].sum() / len(sub) * 100 if "agreed" in sub else 0
    avg_rounds = sub["rounds"].mean()
    print(f"\n  Condition   : {cond}")
    print(f"  Mean Time   : {s.mean():.2f} ms  |  Std: {s.std():.2f} ms")
    print(f"  Median      : {s.median():.2f} ms  |  P95: {np.percentile(s, 95):.2f} ms")
    print(f"  IQR range   : [{q1 - 1.5*iqr:.2f}, {q3 + 1.5*iqr:.2f}] ms")
    print(f"  Avg Rounds  : {avg_rounds:.2f}")
    print(f"  Agreed %    : {agreed_pct:.1f}%")
    print(f"  Anomalies   : {anom} / {len(s)} ({anom/len(s)*100:.1f}%)")

# ─── Plots ────────────────────────────────────────────────────────────────────
x   = np.arange(len(CONDITIONS))
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Negotiation Pattern — Network Analysis & Anomaly Detection",
             fontsize=14, fontweight="bold")

# ── 1. Mean total time bar ────────────────────────────────────────────────────
ax    = axes[0, 0]
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
ax       = axes[0, 1]
box_data = [df[df["network_condition"] == c]["total_time_ms"].values for c in CONDITIONS]
bp       = ax.boxplot(box_data, patch_artist=True,
                      medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], cols):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_xticklabels(CONDITIONS, rotation=20, fontsize=8)
ax.set_title("Total Time Distribution")
ax.set_ylabel("ms")
ax.grid(axis="y", alpha=0.3)

# ── 3. Agreement rate per condition ───────────────────────────────────────────
ax         = axes[0, 2]
agreed_pct = [
    df[df["network_condition"] == c]["agreed"].sum() /
    len(df[df["network_condition"] == c]) * 100
    for c in CONDITIONS
]
bars = ax.bar(CONDITIONS, agreed_pct, color=cols, alpha=0.85)
ax.set_title("Agreement Rate by Condition (%)")
ax.set_ylabel("% Agreed")
ax.tick_params(axis="x", rotation=20)
ax.set_ylim(0, 110)
ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(agreed_pct):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8)

# ── 4. Avg rounds per condition ───────────────────────────────────────────────
ax         = axes[1, 0]
avg_rounds = [df[df["network_condition"] == c]["rounds"].mean() for c in CONDITIONS]
ax.bar(CONDITIONS, avg_rounds, color=cols, alpha=0.85)
ax.set_title("Avg Rounds to Conclude by Condition")
ax.set_ylabel("Rounds")
ax.tick_params(axis="x", rotation=20)
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(avg_rounds):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

# ── 5. Total time by prompt type × condition ──────────────────────────────────
ax      = axes[1, 1]
ptypes  = df["prompt_type"].unique()
width   = 0.8 / len(ptypes)
pcolors = ["#3498db", "#e74c3c", "#2ecc71"]
for i, pt in enumerate(ptypes):
    means_p = [
        df[(df["network_condition"] == c) & (df["prompt_type"] == pt)]["total_time_ms"].mean()
        for c in CONDITIONS
    ]
    ax.bar(x + i * width, means_p, width, label=pt,
           color=pcolors[i % len(pcolors)], alpha=0.85)
ax.set_xticks(x + width * (len(ptypes) - 1) / 2)
ax.set_xticklabels(CONDITIONS, rotation=20, fontsize=8)
ax.set_title("Total Time by Prompt Type × Condition")
ax.set_ylabel("ms")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

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
plt.savefig("/home/jovyan/multi_agent_benchmark/analysis/negotiation_analysis.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved → analysis/negotiation_analysis.png")