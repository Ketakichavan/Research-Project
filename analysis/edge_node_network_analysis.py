import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ─── Load ─────────────────────────────────────────────────────────────────────
CSV_PATH = "/home/jovyan/multi_agent_benchmark/benchmarks/results/edge_node_benchmark.csv"
df = pd.read_csv(CSV_PATH)

# ─── Clean columns ────────────────────────────────────────────────────────────
df["rtt_ms"]           = pd.to_numeric(df["rtt_ms"],           errors="coerce")
df["llm_time_ms"]      = pd.to_numeric(df["llm_time_ms"],      errors="coerce")
df["total_exchanges"]  = pd.to_numeric(df["total_exchanges"],  errors="coerce")
df["measured_latency"] = pd.to_numeric(df["measured_latency"], errors="coerce")
df["measured_jitter"]  = pd.to_numeric(df["measured_jitter"],  errors="coerce")
df["rounds"]           = pd.to_numeric(df["rounds"],           errors="coerce")

# ─── Build unified latency column ─────────────────────────────────────────────
# coordinator uses rtt_ms, p2p and negotiation don't have rtt_ms
# for p2p/negotiation we estimate from measured_latency as proxy
df["latency"] = np.where(
    df["pattern"] == "coordinator",
    df["rtt_ms"],
    df["measured_latency"]
)
df.dropna(subset=["latency"], inplace=True)

# ─── Filter successful runs only ──────────────────────────────────────────────
df = df[df["status"].isin(["success", np.nan]) |
        df["status"].isna()].copy()

EDGE_NODES = [n for n in [
    "aws_london", "aws_frankfurt", "aws_paris",
    "aws_us_east", "aws_tokyo",
    "azure_north_eu", "azure_us_east"
] if n in df["edge_node"].unique()]

COLORS = ["#2ecc71", "#3498db", "#9b59b6",
          "#e67e22", "#e74c3c", "#1abc9c", "#f39c12"]
CMAP   = dict(zip(EDGE_NODES, COLORS))
cols   = [CMAP[n] for n in EDGE_NODES]

PATTERNS = df["pattern"].unique()
PCOLORS  = {"coordinator": "#3498db",
            "p2p":         "#2ecc71",
            "negotiation": "#e74c3c"}

# ─── Anomaly Detection (per edge node per pattern) ───────────────────────────
df["is_anomaly"] = False
for node in EDGE_NODES:
    for pattern in PATTERNS:
        idx = df[(df["edge_node"] == node) &
                 (df["pattern"] == pattern)].index
        if len(idx) < 5:
            continue
        s      = df.loc[idx, "latency"]
        if s.std() < 1:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr    = q3 - q1
        z      = np.abs(stats.zscore(s))
        df.loc[idx, "is_anomaly"] = (
            (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr) | (z > 3)
        )

# ─── Print Summary ────────────────────────────────────────────────────────────
print("=" * 65)
print("EDGE NODE BENCHMARK — ANALYSIS")
print("=" * 65)
for node in EDGE_NODES:
    sub    = df[df["edge_node"] == node]
    s      = sub["latency"]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr    = q3 - q1
    anom   = sub[sub["is_anomaly"]].shape[0]
    meas   = sub["measured_latency"].mean()
    print(f"\n  Edge Node      : {node}")
    print(f"  Provider       : {sub['provider'].iloc[0]} "
          f"| Region: {sub['region'].iloc[0]}")
    print(f"  Measured Ping  : {meas:.2f} ms")
    print(f"  Mean Latency   : {s.mean():.2f} ms  "
          f"|  Std: {s.std():.2f} ms")
    print(f"  Median         : {s.median():.2f} ms  "
          f"|  P95: {np.percentile(s, 95):.2f} ms")
    print(f"  IQR range      : [{q1 - 1.5*iqr:.2f}, "
          f"{q3 + 1.5*iqr:.2f}] ms")
    print(f"  Anomalies      : {anom} / {len(s)} "
          f"({anom/len(s)*100:.1f}%)")

# ─── Plots ────────────────────────────────────────────────────────────────────
x     = np.arange(len(EDGE_NODES))
width = 0.8 / len(PATTERNS)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Edge Node Benchmark — Real AWS/Azure Network Analysis",
             fontsize=14, fontweight="bold")

# ── 1. Mean latency per pattern per edge node ─────────────────────────────────
ax = axes[0, 0]
for i, pattern in enumerate(PATTERNS):
    means = [
        df[(df["edge_node"] == n) &
           (df["pattern"] == pattern)]["latency"].mean()
        for n in EDGE_NODES
    ]
    ax.bar(x + i * width, means, width,
           label=pattern,
           color=PCOLORS.get(pattern, "#aaa"), alpha=0.85)
ax.set_xticks(x + width * (len(PATTERNS) - 1) / 2)
ax.set_xticklabels(EDGE_NODES, rotation=20, fontsize=7)
ax.set_title("Mean Latency per Pattern × Edge Node")
ax.set_ylabel("ms")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(bottom=0)

# ── 2. Box plot per edge node ─────────────────────────────────────────────────
ax       = axes[0, 1]
box_data = [df[df["edge_node"] == n]["latency"].dropna().values
            for n in EDGE_NODES]
bp       = ax.boxplot(box_data, patch_artist=True,
                      medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], cols):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_xticklabels(EDGE_NODES, rotation=20, fontsize=7)
ax.set_title("Latency Distribution per Edge Node")
ax.set_ylabel("ms")
ax.grid(axis="y", alpha=0.3)

# ── 3. Coordinator: RTT by model per edge node ────────────────────────────────
ax    = axes[0, 2]
coord = df[(df["pattern"] == "coordinator") &
           (df["rtt_ms"].notna())].copy()
if not coord.empty:
    models  = coord["model"].dropna().unique()
    mcolors = ["#1abc9c", "#e74c3c", "#f39c12"]
    mwidth  = 0.8 / len(models)
    for i, model in enumerate(models):
        means_m = [
            coord[(coord["edge_node"] == n) &
                  (coord["model"] == model)]["rtt_ms"].mean()
            for n in EDGE_NODES
        ]
        ax.bar(x + i * mwidth, means_m, mwidth,
               label=model,
               color=mcolors[i % len(mcolors)], alpha=0.85)
    ax.set_xticks(x + mwidth * (len(models) - 1) / 2)
    ax.set_xticklabels(EDGE_NODES, rotation=20, fontsize=7)
ax.set_title("Coordinator RTT by Model × Edge Node")
ax.set_ylabel("ms")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(bottom=0)

# ── 4. RTT by prompt type per edge node ───────────────────────────────────────
ax     = axes[1, 0]
ptypes = df["prompt_type"].dropna().unique()
pwidth = 0.8 / len(ptypes)
tcolors = ["#3498db", "#e74c3c", "#2ecc71"]
for i, pt in enumerate(ptypes):
    means_p = [
        df[(df["edge_node"] == n) &
           (df["prompt_type"] == pt)]["latency"].mean()
        for n in EDGE_NODES
    ]
    ax.bar(x + i * pwidth, means_p, pwidth,
           label=pt,
           color=tcolors[i % len(tcolors)], alpha=0.85)
ax.set_xticks(x + pwidth * (len(ptypes) - 1) / 2)
ax.set_xticklabels(EDGE_NODES, rotation=20, fontsize=7)
ax.set_title("Latency by Prompt Type × Edge Node")
ax.set_ylabel("ms")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(bottom=0)

# ── 5. Measured ping vs agent overhead (stacked bar) ─────────────────────────
ax = axes[1, 1]
mean_ping  = [df[df["edge_node"] == n]["measured_latency"].mean()
              for n in EDGE_NODES]
mean_agent = [df[df["edge_node"] == n]["rtt_ms"].mean()
              for n in EDGE_NODES]
overhead   = [max(0, a - p) if not np.isnan(a) else 0
              for a, p in zip(mean_agent, mean_ping)]
ax.bar(EDGE_NODES, mean_ping,
       color="#3498db", alpha=0.85, label="Measured Ping (ms)")
ax.bar(EDGE_NODES, overhead,
       bottom=mean_ping,
       color="#e74c3c", alpha=0.85, label="Agent Overhead (ms)")
ax.set_title("Real Edge Ping vs Agent Overhead")
ax.set_ylabel("ms")
ax.tick_params(axis="x", rotation=20, labelsize=7)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(bottom=0)

# ── 6. Anomaly scatter ────────────────────────────────────────────────────────
ax = axes[1, 2]
for node, color in zip(EDGE_NODES, cols):
    normal = df[(df["edge_node"] == node) & (~df["is_anomaly"])]
    anom   = df[(df["edge_node"] == node) &  (df["is_anomaly"])]
    ax.scatter([node] * len(normal), normal["latency"],
               color=color, alpha=0.4, s=25)
    ax.scatter([node] * len(anom), anom["latency"],
               color="red", marker="x", s=90, linewidths=2)
ax.set_title("Anomalies (Red ✕)")
ax.set_ylabel("Latency (ms)")
ax.tick_params(axis="x", rotation=20, labelsize=7)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("/home/jovyan/multi_agent_benchmark/analysis/edge_node_analysis.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved → analysis/edge_node_analysis.png")