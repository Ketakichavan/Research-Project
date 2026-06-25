# Multi-Agent LLM Coordination Under Network Constraints

> Benchmarking Coordinator-Worker, Peer-to-Peer, and Negotiation patterns for edge-deployed LLM agents under realistic and simulated network impairments.


---

<!-- INSERT dashboard.png HERE -->
*Complete benchmark dashboard — mean latency, anomaly detection, edge suitability scorecard, and AWS/Azure validation across all three coordination patterns and five network conditions.*

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [System Architecture](#system-architecture)
- [The Three Agent Patterns](#the-three-agent-patterns)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Anomaly Detection](#anomaly-detection)
- [Real-World Edge Validation](#real-world-edge-validation)
- [Edge Suitability](#edge-suitability--design-recommendations)
- [Project Structure](#project-structure)
- [How to Reproduce](#how-to-reproduce)
- [Limitations](#limitations--future-work)
- [Authors](#authors--acknowledgments)

---

## Overview

Agentic AI systems built from multiple communicating LLMs are increasingly proposed for edge computing — autonomous vehicles, IoT gateways, and smart factories. But edge networks are inherently unreliable: high latency, jitter, and packet loss are the norm, not the exception. This project implements three fundamental agent communication patterns (Coordinator-Worker, Peer-to-Peer, and Negotiation) using a heterogeneous fleet of local LLMs (Mistral 7B for text, LLaVA for vision), and benchmarks each pattern under five simulated network conditions ranging from ideal baseline to extreme IoT edge links — plus validation against real measured latency to seven AWS/Azure cloud regions. The result is a 250+ run dataset with full statistical analysis, anomaly detection, and an edge-suitability classification for each pattern and condition.

---

## Motivation

Existing research on multi-agent LLM frameworks (AutoGen, CrewAI, LangChain) focuses almost entirely on cloud deployments with stable, low-latency networking. As these systems move to the edge — where connectivity is intermittent and bandwidth-constrained — it remains unclear which coordination patterns stay viable. Does a negotiation protocol that takes 8 seconds in ideal conditions become unusable at 83 seconds under extreme IoT latency? Does peer-to-peer collaboration degrade gracefully or collapse under packet loss? This project answers these questions empirically with a reproducible benchmark suite.

---

## System Architecture

Each "edge node" is simulated as an independent FastAPI process running on its own port (8001–8003), communicating over HTTP on the loopback interface. Network impairments are injected at the Linux kernel level using `tc netem`, affecting all inter-agent TCP traffic identically to a real degraded link. The fleet is heterogeneous — two agents run Mistral 7B (text-only) and one runs LLaVA (vision-language) — mirroring real edge deployments where different devices have different AI capabilities.

<!-- INSERT architecture diagram HERE -->

| Layer | Technology |
|---|---|
| LLM Inference | Ollama (local), Mistral 7B + LLaVA |
| Agent Servers | FastAPI + Uvicorn (Python) |
| Inter-agent Transport | HTTP over loopback (127.0.0.1) |
| Network Impairment | Linux `tc netem` (kernel-level) |
| Hardware | NVIDIA GH200 144GB HBM3e (MIG 1g.18gb), ARM64 |
| Logging | JSONL structured event log |
| Analysis | pandas, matplotlib, scipy |

---

## The Three Agent Patterns

### Pattern 1 — Coordinator-Worker

A central coordinator receives a task and uses Mistral to decompose it into three subtasks. Each subtask is dispatched to one of three worker agents on dedicated ports, who process it using their assigned model (Mistral or LLaVA) and return a result. This models task delegation — a central orchestrator distributing work across heterogeneous edge nodes.

```
Coordinator (Mistral)
    │ breaks task into 3 subtasks
    ├──► Worker 1 (port 8001, Mistral) → processes subtask 1
    ├──► Worker 2 (port 8002, Mistral) → processes subtask 2
    └──► Worker 3 (port 8003, LLaVA)  → processes subtask 3
         ↑ image prompts routed here only
```

### Pattern 2 — Peer-to-Peer

Two agents engage in a symmetric multi-round discussion (3 rounds) with no hierarchy. Each agent generates a response, sends it to its peer via HTTP, receives the peer's contribution, and builds on the combined conversation history for the next round. All three peer pair combinations are tested: Mistral↔Mistral, Mistral↔LLaVA, and LLaVA↔Mistral.

```
peer_agent_1 ──── /p2p_respond ────► peer_agent_2
     ▲                                     │
     └──────────── response ───────────────┘
          (repeated for 3 rounds)
```

### Pattern 3 — Negotiation

Agent 1 (Mistral, initiator) proposes a solution. Responding agents (Mistral + LLaVA) each independently evaluate and return either `AGREED:` or `COUNTER:`. If any agent agrees, negotiation ends and logs which agent/round reached agreement. Otherwise the initiator refines its proposal based on all feedback and tries again, up to 4 rounds.

```
agent_1 (Mistral) ── proposal ──► agent_2 (Mistral): AGREED or COUNTER
                               ──► agent_3 (LLaVA):  AGREED or COUNTER
     ▲                                     │
     └──── refined proposal ◄── feedback ──┘
          (up to 4 rounds)
```

---

## Experimental Setup

### Test Workload

Each pattern was tested against an identical 8-prompt workload across three categories:

| Type | Count | Example |
|---|---|---|
| Simple | 3 | "What is latency in networking?" |
| Complex | 2 | "Design a fault-tolerant edge architecture for an autonomous vehicle handling sensor failures, network partitions, and real-time decisions simultaneously." |
| Image | 3 | "Describe this network topology diagram and identify potential bottlenecks." (LLaVA only) |

> Image prompts are **skipped for Mistral workers** — a text-only model cannot process images and would hallucinate content about something it cannot see.

<!-- INSERT benchmarks/images/network_topology.png HERE -->
*Conceptual edge network topology used for image-based prompt testing — IoT devices connected through edge nodes to cloud, analyzed by the LLaVA vision agent.*

### Network Conditions

| Condition | `tc netem` Command | Represents |
|---|---|---|
| `baseline` | none | Ideal local network |
| `100ms_delay` | `delay 100ms` | Moderate WAN / edge link |
| `jitter_50ms_gaussian` | `delay 100ms 50ms distribution normal` | Variable wireless link (4G/5G) |
| `packet_loss_5percent` | `loss 5%` | Congested network |
| `iot_edge_extreme` | `delay 500ms loss 2% rate 256kbit` | Worst-case IoT / satellite link |

### Dataset Summary

| Pattern | Total Runs | Description |
|---|---|---|
| Coordinator-Worker | 90 | 5 conditions × 18 worker-runs |
| Peer-to-Peer | 120 | 5 conditions × 24 pair-runs (3 pairs × 8 prompts) |
| Negotiation | 40 | 5 conditions × 8 prompt-runs |

---

## Results

### Coordinator-Worker

| Condition | Mistral avg RTT | LLaVA avg RTT |
|---|---|---|
| baseline | 2,197 ms | 20,520 ms |
| 100ms_delay | 3,377 ms | 11,775 ms |
| jitter_50ms_gaussian | 3,518 ms | 2,841 ms |
| packet_loss_5percent | 1,961 ms | 1,766 ms |
| iot_edge_extreme | 11,435 ms | 10,007 ms |

Mistral shows a clear expected degradation as latency increases. LLaVA's baseline is dramatically higher (~20s) due to vision encoder overhead, but counterintuitively improves under jitter and loss conditions — its bottleneck is compute-bound rather than network-bound at moderate impairment levels. Under extreme IoT conditions both models converge to similarly poor performance (~10–11s).

<!-- INSERT analysis/coordinator_analysis.png HERE -->
*Coordinator-Worker analysis — mean RTT by condition, distribution box plots, Mistral vs LLaVA comparison, prompt type breakdown, LLM time vs network overhead, and anomaly scatter plot.*

---

### Peer-to-Peer

| Condition | Mean Total Time |
|---|---|
| baseline | 25,205 ms |
| 100ms_delay | 24,870 ms |
| jitter_50ms_gaussian | 25,447 ms |
| packet_loss_5percent | 19,119 ms |
| iot_edge_extreme | 71,140 ms |

P2P is the most resilient pattern. Baseline, 100ms delay, and jitter all cluster around 25 seconds — varying by less than 2.3%. Performance even improves under packet loss (19.1s) because dropped packets have negligible impact relative to the LLM generation time that dominates total duration. Only the extreme IoT condition (500ms + loss + bandwidth cap) causes severe degradation (+182% vs baseline).

<!-- INSERT analysis/p2p_analysis.png HERE -->
*Peer-to-Peer analysis — showing strong resilience across moderate conditions and sharp degradation only under combined extreme IoT impairments.*

---

### Negotiation

| Condition | Mean Total Time | Avg Rounds | Agreement Rate |
|---|---|---|---|
| baseline | 7,957 ms | 2.0 | 100% |
| 100ms_delay | 21,266 ms | 2.9 | 75% |
| jitter_50ms_gaussian | 18,635 ms | 2.1 | 75% |
| packet_loss_5percent | 9,349 ms | 2.1 | 88% |
| iot_edge_extreme | 82,952 ms | 2.8 | 88% |

Negotiation is the most network-sensitive pattern. Under 100ms delay alone, total time increases by 167% and agreement rate drops from 100% to 75% — meaning one in four negotiations fails to converge within the round limit when latency is introduced. This compounding effect occurs because each round adds the full network penalty for every participating agent. Under extreme IoT conditions, negotiation takes over 83 seconds — completely impractical for real-time edge deployment.

<!-- INSERT analysis/negotiation_analysis.png HERE -->
*Negotiation analysis — total time per condition, agreement rate, average rounds to convergence, prompt type breakdown, and anomaly detection.*

---

## Anomaly Detection

Every pattern's results were screened for statistically unusual response times using a combined approach:

**IQR (Interquartile Range) 1.5× Rule:**
- Compute Q1 (25th percentile) and Q3 (75th percentile)
- IQR = Q3 − Q1
- Flag anything below `Q1 − 1.5×IQR` or above `Q3 + 1.5×IQR`

**Z-Score threshold (|z| > 3):**
- Flag any reading more than 3 standard deviations from the mean

A reading is flagged as anomalous if **either** method catches it.

**Critical design decision — per-group detection:** Anomaly detection is computed independently per model per condition (coordinator) or per condition (P2P/negotiation). This prevents LLaVA's structurally higher latency from making all Mistral readings appear anomalously fast by comparison. Each group's "normal" is defined relative to its own distribution.

Flagged anomalies (shown as red ✕ markers in analysis plots) are candidates for cross-correlation with packet-level tcpdump captures — an application-level anomaly may correspond to a TCP retransmission burst, timeout event, or bandwidth saturation detectable purely from network traffic.

---

## Real-World Edge Validation

Beyond simulated `tc netem` conditions, live latency was measured against seven real cloud endpoints:

| Provider | Regions |
|---|---|
| AWS | London, Frankfurt, Paris, US-East, Tokyo |
| Azure | North Europe, US-East |

Each endpoint was probed 15 times (200ms apart) to measure mean latency, jitter (standard deviation), and packet loss. These real measurements were then translated into corresponding `tc netem` commands and all three patterns were re-run under those real-world-derived conditions — grounding the simulation in actual measured internet characteristics.

<!-- INSERT analysis/edge_node_analysis.png HERE -->
*Real AWS/Azure edge node analysis — latency per pattern per region, distribution box plots, model comparison, measured ping vs agent overhead, and anomaly scatter.*

---

## Edge Suitability & Design Recommendations

Based on standard edge computing latency requirements:

| Tier | Threshold | Use case |
|---|---|---|
| RT (Real-Time) | < 500 ms | Control loops, safety-critical |
| IA (Interactive) | < 5,000 ms | User-facing applications |
| BA (Batch) | < 30,000 ms | Background processing |
| FAIL | > 30,000 ms | Unsuitable for edge |

**Suitability scorecard across patterns and conditions:**

| Pattern | Baseline | 100ms | Jitter | Loss | IoT Extreme |
|---|---|---|---|---|---|
| Coordinator-Worker | IA | IA | IA | IA | FAIL |
| Peer-to-Peer | BA | BA | BA | BA | FAIL |
| Negotiation | IA | FAIL | FAIL | IA | FAIL |

**Design recommendations:**

Coordinator-Worker is the most edge-friendly pattern overall, remaining in the interactive tier across four of five conditions and offering the clearest separation between LLM inference time and network overhead — making it the recommended default for edge-deployed multi-agent systems. Negotiation should be avoided in any environment with more than 100ms latency unless adaptive round limits or timeout-based fallback mechanisms are implemented, as its multi-round structure causes latency to compound multiplicatively rather than additively. Peer-to-Peer is recommended for environments expecting moderate jitter or packet loss due to its demonstrated resilience, but requires circuit-breaker logic for extreme IoT conditions where combined impairments (high latency + loss + bandwidth cap) cause total session time to exceed 70 seconds.

---

## Project Structure

```
multi_agent_benchmark/
│
├── core/
│   ├── llm.py                    # Ollama API wrapper (text + image/base64 support)
│   ├── message.py                # Pydantic message schema for request validation
│   └── logger.py                 # Structured JSONL event logger
│
├── agents/
│   ├── coordinator.py            # Pattern 1: Coordinator-Worker
│   ├── peer_agent.py             # Pattern 2: Peer-to-Peer (3 pairs)
│   ├── negotiation_agent.py      # Pattern 3: Multi-agent Negotiation
│   └── worker.py                 # FastAPI edge-node server (/process, /p2p_respond, /negotiate)
│
├── network/
│   └── impairments.sh            # tc netem helper (apply_latency, apply_loss, reset)
│
├── benchmarks/
│   ├── prompts.py                # Standardized 8-prompt workload (simple/complex/image)
│   ├── run_benchmark.py          # Coordinator experiment runner
│   ├── run_p2p_benchmark.py      # P2P experiment runner
│   ├── run_negotiation_benchmark.py  # Negotiation experiment runner
│   ├── images/                   # Auto-generated test diagrams for LLaVA
│   └── results/                  # Raw CSV + JSON datasets
│       ├── coordinator_benchmark.csv
│       ├── p2p_results_benchmark.csv
│       └── benchmark_negotiation_results.csv
│
├── analysis/
│   ├── coordinator_network_Analysis.py   # Coordinator anomaly detection + plots
│   ├── p2p_network_analysis.py           # P2P anomaly detection + plots
│   ├── negotiation_network_analysis.py   # Negotiation anomaly detection + plots
│   ├── edge_node_network_analysis.py     # Real AWS/Azure edge node analysis
│   ├── edge_node_traffic_capture.py      # Live edge node latency measurement + replay
│   └── dashboard.py                      # Unified 10-panel summary dashboard
│
└── start_workers.sh              # Launches all 3 workers on ports 8001–8003
```

---

## How to Reproduce

### Prerequisites

```bash
# Install Ollama and pull models
ollama pull mistral
ollama pull llava

# Install Python dependencies
pip install fastapi uvicorn requests pandas matplotlib scipy pydantic
```

### Start the Agent Fleet

```bash
# Start all 3 workers (Mistral × 2, LLaVA × 1)
./start_workers.sh

# Or manually:
python3 agents/worker.py worker_1 8001 &
python3 agents/worker.py worker_2 8002 &
python3 agents/worker.py worker_3 8003 &
```

### Run Benchmarks

```bash
# Pattern 1 — Coordinator-Worker
python3 benchmarks/run_benchmark.py

# Pattern 2 — Peer-to-Peer
python3 benchmarks/run_p2p_benchmark.py

# Pattern 3 — Negotiation
python3 benchmarks/run_negotiation_benchmark.py

# Real AWS/Azure edge validation
python3 analysis/edge_node_traffic_capture.py
```

### Generate Analysis

```bash
python3 analysis/coordinator_network_Analysis.py
python3 analysis/p2p_network_analysis.py
python3 analysis/negotiation_network_analysis.py
python3 analysis/edge_node_network_analysis.py
python3 analysis/dashboard.py
```

Results and plots saved to `analysis/` and `benchmarks/results/`.

### Quick Single Test

```bash
# Test coordinator under baseline
python3 agents/coordinator.py "Explain edge computing for IoT" baseline

# Test with 200ms delay
sudo tc qdisc add dev lo root netem delay 200ms
python3 agents/coordinator.py "Explain edge computing for IoT" 200ms_delay
sudo tc qdisc del dev lo root
```

---

## Limitations & Future Work

**Known limitations:**

- All experiments run on a single physical machine — processes share one GPU for inference rather than having dedicated compute per node as in a real distributed deployment
- The loopback interface has near-zero baseline latency; real physical networks have 1–5ms baseline overhead even in ideal conditions
- Sequential benchmark execution (required due to Ollama's single inference queue) means agents cannot exhibit true parallel behavior
- Negotiation pattern has 40 runs per condition — more repetitions would strengthen statistical confidence in anomaly detection

**Future work:**

- Deploy on actual distributed hardware (e.g., Raspberry Pi 5 cluster) to eliminate shared-GPU limitation
- Increase repetitions per condition to ≥ 30 for stronger statistical power
- Implement streaming mode comparison — token-by-token responses may remain partially useful even if the connection drops under extreme IoT conditions
- Integrate packet-capture-based anomaly detection to cross-correlate application-layer RTT anomalies with network-layer events (TCP retransmissions, timeout bursts)
- Test additional model sizes to study how model complexity interacts with network conditions at the edge

---

