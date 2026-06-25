import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ─── Load all data ────────────────────────────────────────────────────────────
BASE = "/home/jovyan/multi_agent_benchmark/benchmarks/results"

# Coordinator
df_coord = pd.read_csv(f"{BASE}/coordinator_benchmark.csv")
df_coord = df_coord[df_coord["status"] == "success"].copy()
df_coord["rtt_ms"]      = pd.to_numeric(df_coord["rtt_ms"],      errors="coerce")
df_coord["llm_time_ms"] = pd.to_numeric(df_coord["llm_time_ms"], errors="coerce")
df_coord["pattern"]     = "coordinator"

# P2P
df_p2p = pd.read_csv(f"{BASE}/p2p_results_benchmark.csv")
df_p2p["total_time_ms"] = pd.to_numeric(df_p2p["total_time_ms"], errors="coerce")
df_p2p["pattern"]       = "p2p"

# Negotiation
df_neg = pd.read_csv(f"{BASE}/benchmark_negotiation_results.csv")
df_neg["total_time_ms"] = pd.to_numeric(df_neg["total_time_ms"], errors="coerce")
df_neg["agreed"]        = df_neg["agreed"].astype(bool)
df_neg["rounds"]        = pd.to_numeric(df_neg["rounds"],        errors="coerce")
df_neg["pattern"]       = "negotiation"

# Edge node benchmark
df_edge = pd.read_csv(f"{BASE}/edge_node_benchmark.csv")
df_edge["rtt_ms"]           = pd.to_numeric(df_edge["rtt_ms"],           errors="coerce")
df_edge["measured_latency"] = pd.to_numeric(df_edge["measured_latency"], errors="coerce")
df_edge["latency"]          = df_edge["rtt_ms"].combine_first(
                                  df_edge["measured_latency"])

# ─── Shared constants ─────────────────────────────────────────────────────────
SIM_CONDITIONS = [c for c in [
    "baseline", "100ms_delay", "jitter_50ms_gaussian",
    "packet_loss_5percent", "iot_edge_extreme"
] if c in df_coord["network_condition"].unique()]

EDGE_NODES = [n for n in [
    "aws_london", "aws_frankfurt", "aws_paris",
    "aws_us_east", "aws_tokyo",
    "azure_north_eu", "azure_us_east"
] if n in df_edge["edge_node"].unique()]

PATTERNS  = ["coordinator", "p2p", "negotiation"]
PCOLORS   = {"coordinator": "#3498db",
             "p2p":         "#2ecc71",
             "negotiation": "#e74c3c"}
CCOLORS   = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c"]
CMAP      = dict(zip(SIM_CONDITIONS, CCOLORS))
ECCOLORS  = ["#2ecc71", "#3498db", "#9b59b6",
             "#e67e22", "#e74c3c", "#1abc9c", "#f39c12"]
ECMAP     = dict(zip(EDGE_NODES, ECCOLORS))

# ─── Anomaly detection helper ─────────────────────────────────────────────────
def flag_anomalies(series):
    if len(series) < 5 or series.std() < 1:
        return pd.Series(False, index=series.index)
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr    = q3 - q1
    z      = np.abs(stats.zscore(series))
    return (series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr) | (z > 3)

# ─── Build dashboard ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 22))
fig.patch.set_facecolor("#f8f9fa")
gs  = gridspec.GridSpec(4, 3, figure=fig,
                        hspace=0.65, wspace=0.35)
fig.suptitle(
    "Multi-Agent Benchmark — Complete Dashboard\n"
    "Simulated Conditions · Edge Suitability · Real AWS/Azure Results",
    fontsize=15, fontweight="bold", y=0.98      # ← fixed white space
)

x     = np.arange(len(SIM_CONDITIONS))
width = 0.8 / len(PATTERNS)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Simulated benchmark overview
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Mean latency all 3 patterns × simulated condition ─────────────────────
ax = fig.add_subplot(gs[0, :2])
for i, pattern in enumerate(PATTERNS):
    if pattern == "coordinator":
        means = [df_coord[df_coord["network_condition"] == c]["rtt_ms"].mean()
                 for c in SIM_CONDITIONS]
    elif pattern == "p2p":
        means = [df_p2p[df_p2p["network_condition"] == c]["total_time_ms"].mean()
                 for c in SIM_CONDITIONS]
    else:
        means = [df_neg[df_neg["network_condition"] == c]["total_time_ms"].mean()
                 for c in SIM_CONDITIONS]
    ax.bar(x + i * width, means, width,
           label=pattern,
           color=PCOLORS[pattern], alpha=0.85)
ax.set_xticks(x + width)
ax.set_xticklabels(SIM_CONDITIONS, rotation=15, fontsize=8)
ax.set_title("Mean Latency — All Patterns x Simulated Conditions",
             fontweight="bold")
ax.set_ylabel("ms")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(bottom=0)
ax.set_facecolor("#ffffff")

# ── 2. Degradation % vs baseline ─────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
for pattern in PATTERNS:
    if pattern == "coordinator":
        base = df_coord[df_coord["network_condition"] == "baseline"]["rtt_ms"].mean()
        vals = [df_coord[df_coord["network_condition"] == c]["rtt_ms"].mean()
                for c in SIM_CONDITIONS]
    elif pattern == "p2p":
        base = df_p2p[df_p2p["network_condition"] == "baseline"]["total_time_ms"].mean()
        vals = [df_p2p[df_p2p["network_condition"] == c]["total_time_ms"].mean()
                for c in SIM_CONDITIONS]
    else:
        base = df_neg[df_neg["network_condition"] == "baseline"]["total_time_ms"].mean()
        vals = [df_neg[df_neg["network_condition"] == c]["total_time_ms"].mean()
                for c in SIM_CONDITIONS]
    pct = [(v - base) / base * 100 for v in vals]
    ax.plot(SIM_CONDITIONS, pct, marker="o",
            label=pattern, color=PCOLORS[pattern],
            linewidth=2, markersize=5)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Degradation vs Baseline (%)", fontweight="bold")
ax.set_ylabel("% increase")
ax.tick_params(axis="x", rotation=15, labelsize=7)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
ax.set_facecolor("#ffffff")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Pattern specific findings
# ══════════════════════════════════════════════════════════════════════════════

# ── 3. Coordinator: mistral vs llava ─────────────────────────────────────────
ax    = fig.add_subplot(gs[1, 0])
xc    = np.arange(len(SIM_CONDITIONS))
wc    = 0.8 / 2
for i, (model, color) in enumerate(
        zip(["mistral", "llava"], ["#1abc9c", "#e74c3c"])):
    means = [
        df_coord[(df_coord["network_condition"] == c) &
                 (df_coord["model"] == model)]["rtt_ms"].mean()
        for c in SIM_CONDITIONS
    ]
    ax.bar(xc + i * wc, means, wc,
           label=model, color=color, alpha=0.85)
ax.set_xticks(xc + wc / 2)
ax.set_xticklabels(SIM_CONDITIONS, rotation=15, fontsize=7)
ax.set_title("Coordinator\nMistral vs LLaVA", fontweight="bold")
ax.set_ylabel("RTT (ms)")
ax.legend(fontsize=7)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(bottom=0)
ax.set_facecolor("#ffffff")

# ── 4. Negotiation: agreement rate + avg rounds ───────────────────────────────
ax  = fig.add_subplot(gs[1, 1])
ax2 = ax.twinx()
agreed_pct = [
    df_neg[df_neg["network_condition"] == c]["agreed"].sum() /
    len(df_neg[df_neg["network_condition"] == c]) * 100
    for c in SIM_CONDITIONS
]
avg_rounds = [
    df_neg[df_neg["network_condition"] == c]["rounds"].mean()
    for c in SIM_CONDITIONS
]
ax.bar(SIM_CONDITIONS, agreed_pct,
       color=[CMAP[c] for c in SIM_CONDITIONS],
       alpha=0.65, label="Agreement %")
ax2.plot(SIM_CONDITIONS, avg_rounds,
         color="black", marker="o",
         linewidth=2, markersize=5, label="Avg Rounds")
ax.set_title("Negotiation Quality\nAgreement % + Rounds",
             fontweight="bold")
ax.set_ylabel("Agreement %")
ax2.set_ylabel("Rounds")
ax.set_ylim(0, 115)
ax2.set_ylim(0, 5)
ax.tick_params(axis="x", rotation=15, labelsize=7)
ax.grid(axis="y", alpha=0.3)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7)
ax.set_facecolor("#ffffff")

# ── 5. P2P: avg exchanges per condition ───────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
ax.bar(SIM_CONDITIONS,
       [df_p2p[df_p2p["network_condition"] == c]["total_exchanges"].mean()
        for c in SIM_CONDITIONS],
       color=[CMAP[c] for c in SIM_CONDITIONS], alpha=0.85)
ax.set_title("P2P\nAvg Exchanges per Condition", fontweight="bold")
ax.set_ylabel("Exchanges")
ax.tick_params(axis="x", rotation=15, labelsize=7)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(bottom=0)
ax.set_facecolor("#ffffff")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Real AWS/Azure edge node results
# ══════════════════════════════════════════════════════════════════════════════

# ── 6. Mean latency per pattern per real edge node ────────────────────────────
ax    = fig.add_subplot(gs[2, :2])
xe    = np.arange(len(EDGE_NODES))
ew    = 0.8 / len(PATTERNS)
ep    = df_edge["pattern"].unique()
for i, pattern in enumerate(ep):
    means = [
        df_edge[(df_edge["edge_node"] == n) &
                (df_edge["pattern"] == pattern)]["latency"].mean()
        for n in EDGE_NODES
    ]
    ax.bar(xe + i * ew, means, ew,
           label=pattern,
           color=PCOLORS.get(pattern, "#aaa"), alpha=0.85)
ax.set_xticks(xe + ew * (len(ep) - 1) / 2)
ax.set_xticklabels(EDGE_NODES, rotation=15, fontsize=7)
ax.set_title("Real AWS/Azure Edge Nodes — Latency per Pattern",
             fontweight="bold")
ax.set_ylabel("ms")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(bottom=0)
ax.set_facecolor("#ffffff")

# ── 7. Real ping vs agent overhead ────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 2])
mean_ping  = [df_edge[df_edge["edge_node"] == n]["measured_latency"].mean()
              for n in EDGE_NODES]
mean_agent = [df_edge[df_edge["edge_node"] == n]["rtt_ms"].mean()
              for n in EDGE_NODES]
overhead   = [max(0, a - p) if not np.isnan(a) else 0
              for a, p in zip(mean_agent, mean_ping)]
ax.bar(EDGE_NODES, mean_ping,
       color="#3498db", alpha=0.85, label="Real Ping")
ax.bar(EDGE_NODES, overhead,
       bottom=mean_ping,
       color="#e74c3c", alpha=0.85, label="Agent Overhead")
ax.set_title("Real Ping vs\nAgent Overhead", fontweight="bold")
ax.set_ylabel("ms")
ax.tick_params(axis="x", rotation=15, labelsize=7)
ax.legend(fontsize=7)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(bottom=0)
ax.set_facecolor("#ffffff")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4 — Edge suitability + anomaly summary
# ══════════════════════════════════════════════════════════════════════════════

# ── 8. Edge suitability scorecard ─────────────────────────────────────────────
ax = fig.add_subplot(gs[3, 0])
ax.axis("off")

RT = 500
IA = 5000
BA = 30000

score_rows = []
for pattern in PATTERNS:
    row = []
    for cond in SIM_CONDITIONS:
        if pattern == "coordinator":
            mean = df_coord[df_coord["network_condition"] == cond]["rtt_ms"].mean()
        elif pattern == "p2p":
            mean = df_p2p[df_p2p["network_condition"] == cond]["total_time_ms"].mean()
        else:
            mean = df_neg[df_neg["network_condition"] == cond]["total_time_ms"].mean()
        if   mean < RT: row.append("RT")        # ← no emoji
        elif mean < IA: row.append("IA")        # ← no emoji
        elif mean < BA: row.append("BA")        # ← no emoji
        else:           row.append("FAIL")      # ← no emoji
    score_rows.append(row)

table = ax.table(
    cellText=score_rows,
    rowLabels=PATTERNS,
    colLabels=["Base", "100ms", "Jitter", "Loss", "IoT"],
    loc="center", cellLoc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1.2, 1.8)
ax.set_title("Edge Suitability Scorecard\n"           # ← no emoji
             "RT=<500ms | IA=<5s | BA=<30s | FAIL=Unsuitable",
             fontweight="bold", fontsize=8, pad=40)

# ── 9. Anomaly summary all patterns ──────────────────────────────────────────
ax = fig.add_subplot(gs[3, 1])

# coordinator anomalies
df_coord["is_anomaly"] = False
for cond in SIM_CONDITIONS:
    for model in df_coord["model"].unique():
        idx = df_coord[(df_coord["network_condition"] == cond) &
                       (df_coord["model"] == model)].index
        if len(idx) >= 5:
            s = df_coord.loc[idx, "rtt_ms"]
            df_coord.loc[idx, "is_anomaly"] = flag_anomalies(s)

# p2p anomalies
df_p2p["is_anomaly"] = False
for cond in SIM_CONDITIONS:
    idx = df_p2p[df_p2p["network_condition"] == cond].index
    if len(idx) >= 5:
        s = df_p2p.loc[idx, "total_time_ms"]
        df_p2p.loc[idx, "is_anomaly"] = flag_anomalies(s)

# negotiation anomalies
df_neg["is_anomaly"] = False
for cond in SIM_CONDITIONS:
    idx = df_neg[df_neg["network_condition"] == cond].index
    if len(idx) >= 5:
        s = df_neg.loc[idx, "total_time_ms"]
        df_neg.loc[idx, "is_anomaly"] = flag_anomalies(s)

anom_data = {
    "coordinator": [df_coord[df_coord["network_condition"] == c]["is_anomaly"].sum()
                    for c in SIM_CONDITIONS],
    "p2p":         [df_p2p[df_p2p["network_condition"] == c]["is_anomaly"].sum()
                    for c in SIM_CONDITIONS],
    "negotiation": [df_neg[df_neg["network_condition"] == c]["is_anomaly"].sum()
                    for c in SIM_CONDITIONS],
}
aw = 0.8 / len(PATTERNS)
for i, pattern in enumerate(PATTERNS):
    ax.bar(x + i * aw, anom_data[pattern], aw,
           label=pattern, color=PCOLORS[pattern], alpha=0.85)
ax.set_xticks(x + aw)
ax.set_xticklabels(SIM_CONDITIONS, rotation=15, fontsize=7)
ax.set_title("Anomaly Count\nAll Patterns x Condition",
             fontweight="bold")
ax.set_ylabel("Count")
ax.legend(fontsize=7)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(bottom=0)
ax.set_facecolor("#ffffff")

# ── 10. AWS vs Azure mean latency ─────────────────────────────────────────────
ax        = fig.add_subplot(gs[3, 2])
providers = ["AWS", "Azure"]
pw        = 0.8 / len(ep)
xprov     = np.arange(len(providers))
for i, pattern in enumerate(ep):
    means = [
        df_edge[(df_edge["provider"] == p) &
                (df_edge["pattern"] == pattern)]["latency"].mean()
        for p in providers
    ]
    ax.bar(xprov + i * pw, means, pw,
           label=pattern,
           color=PCOLORS.get(pattern, "#aaa"), alpha=0.85)
ax.set_xticks(xprov + pw * (len(ep) - 1) / 2)
ax.set_xticklabels(providers, fontsize=9)
ax.set_title("AWS vs Azure\nMean Latency per Pattern",
             fontweight="bold")
ax.set_ylabel("ms")
ax.legend(fontsize=7)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(bottom=0)
ax.set_facecolor("#ffffff")

# ─── Save ─────────────────────────────────────────────────────────────────────
ANALYSIS_DIR = "/home/jovyan/multi_agent_benchmark/analysis"
plt.tight_layout()
plt.savefig(f"{ANALYSIS_DIR}/dashboard.png",
            dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("\nDashboard saved → analysis/dashboard.png")