import requests
import time
import statistics
import json
import os
import sys
import csv
from datetime import datetime

sys.path.insert(0, '/home/jovyan/multi_agent_benchmark')
sys.path.insert(0, '/home/jovyan/multi_agent_benchmark/core')

from agents.negotiation_agent import run_negotiation
from agents.peer_agent        import run_p2p_exchange
from benchmarks.prompts       import SIMPLE_PROMPTS, COMPLEX_PROMPTS, IMAGE_PROMPTS

# ─── Workers (same as coordinator benchmark) ──────────────────────────────────
WORKERS = [
    {"id": "worker_1", "url": "http://127.0.0.1:8001", "model": "mistral"},
    {"id": "worker_2", "url": "http://127.0.0.1:8002", "model": "mistral"},
    {"id": "worker_3", "url": "http://127.0.0.1:8003", "model": "llava"},
]

# ─── Real AWS/Azure Edge Endpoints ───────────────────────────────────────────
EDGE_NODES = {
    "aws_london":     {"url": "https://s3.eu-west-2.amazonaws.com",       "provider": "AWS",   "region": "Europe"},
    "aws_frankfurt":  {"url": "https://cloudfront.amazonaws.com",          "provider": "AWS",   "region": "Europe"},
    "aws_paris":      {"url": "https://s3.eu-west-3.amazonaws.com",        "provider": "AWS",   "region": "Europe"},
    "aws_us_east":    {"url": "https://s3.us-east-1.amazonaws.com",        "provider": "AWS",   "region": "US"},
    "aws_tokyo":      {"url": "https://s3.ap-northeast-1.amazonaws.com",   "provider": "AWS",   "region": "Asia"},
    "azure_north_eu": {"url": "https://northeurope.blob.core.windows.net", "provider": "Azure", "region": "Europe"},
    "azure_us_east":  {"url": "https://eastus.blob.core.windows.net",      "provider": "Azure", "region": "US"},
}

RESET_CMD = "sudo tc qdisc del dev lo root 2>/dev/null; true"

# ─── Measure live latency to a real edge node ─────────────────────────────────
def measure_edge_node(name, node_info, samples=15):
    url       = node_info["url"]
    latencies = []
    failures  = 0

    for _ in range(samples):
        try:
            start = time.time()
            requests.get(url, timeout=3)
            latencies.append((time.time() - start) * 1000)
        except:
            failures += 1
        time.sleep(0.2)

    if not latencies:
        return None

    return {
        "node":        name,
        "provider":    node_info["provider"],
        "region":      node_info["region"],
        "url":         url,
        "mean_ms":     round(statistics.mean(latencies), 2),
        "jitter_ms":   round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0.0,
        "min_ms":      round(min(latencies), 2),
        "max_ms":      round(max(latencies), 2),
        "loss_pct":    round(failures / samples * 100, 1),
        "samples":     samples,
        "measured_at": datetime.now().isoformat(),
    }

# ─── Build tc command from real measured values ───────────────────────────────
def build_tc_from_edge_node(node_stats):
    delay  = max(1, round(node_stats["mean_ms"]))
    jitter = max(0, round(node_stats["jitter_ms"]))
    loss   = node_stats["loss_pct"]

    cmd = f"sudo tc qdisc add dev lo root netem delay {delay}ms"
    if jitter > 5:
        cmd += f" {jitter}ms distribution normal"
    if loss > 0.5:
        cmd += f" loss {loss}%"
    return cmd

def apply_edge_condition(tc_cmd):
    if os.system("which tc 2>/dev/null") == 0:
        os.system(RESET_CMD)
        os.system(tc_cmd)
        time.sleep(0.5)
    else:
        print("    tc not available — edge node stats recorded as metadata only")

def reset_network():
    if os.system("which tc 2>/dev/null") == 0:
        os.system(RESET_CMD)

# ─── Capture all edge nodes right now ────────────────────────────────────────
def capture_all_edge_nodes():
    print("=" * 65)
    print("LIVE EDGE NODE TRAFFIC CAPTURE")
    print("Measuring real AWS/Azure edge node conditions RIGHT NOW")
    print("=" * 65)

    live_nodes = []
    for name, info in EDGE_NODES.items():
        print(f"\n  Probing {name} ({info['provider']} {info['region']})...")
        stats = measure_edge_node(name, info)
        if stats:
            live_nodes.append(stats)
            print(f"  ✓ Mean: {stats['mean_ms']}ms | "
                  f"Jitter: {stats['jitter_ms']}ms | "
                  f"Loss: {stats['loss_pct']}%")
        else:
            print(f"  ✗ Unreachable")

    return live_nodes

# ─── Coordinator: send_to_worker (exact same as coordinator benchmark) ────────
def send_to_worker(worker, prompt_text, image_path,
                   condition_name, prompt_type, prompt_index,
                   node, tc_cmd):
    payload = {
        "sender":            "benchmark_runner",
        "receiver":          worker["id"],
        "content":           prompt_text,
        "round_number":      1,
        "pattern":           "coordinator_worker",
        "network_condition": condition_name,
    }
    if image_path:
        payload["image_path"] = image_path

    start = time.time()
    try:
        r      = requests.post(f"{worker['url']}/process",
                               json=payload, timeout=180)
        rtt_ms = round((time.time() - start) * 1000, 2)
        data   = r.json()
        result = {
            "pattern":           "coordinator",
            "edge_node":         node["node"],
            "provider":          node["provider"],
            "region":            node["region"],
            "measured_latency":  node["mean_ms"],
            "measured_jitter":   node["jitter_ms"],
            "measured_loss_pct": node["loss_pct"],
            "tc_cmd_applied":    tc_cmd,
            "worker_id":         worker["id"],
            "model":             worker["model"],
            "prompt_type":       prompt_type,
            "prompt_index":      prompt_index,
            "prompt_length":     len(prompt_text),
            "has_image":         image_path is not None,
            "rtt_ms":            rtt_ms,
            "response_length":   len(data.get("result", "")),
            "llm_time_ms":       data.get("llm_time_ms", 0),
            "status":            "success",
            "agreed":            None,
            "rounds":            None,
            "total_exchanges":   None,
            "timestamp":         datetime.now().isoformat(),
        }
        print(f"    ✓ {worker['id']} ({worker['model']}) — {rtt_ms}ms")
    except Exception as e:
        rtt_ms = round((time.time() - start) * 1000, 2)
        result = {
            "pattern":           "coordinator",
            "edge_node":         node["node"],
            "provider":          node["provider"],
            "region":            node["region"],
            "measured_latency":  node["mean_ms"],
            "measured_jitter":   node["jitter_ms"],
            "measured_loss_pct": node["loss_pct"],
            "tc_cmd_applied":    tc_cmd,
            "worker_id":         worker["id"],
            "model":             worker["model"],
            "prompt_type":       prompt_type,
            "prompt_index":      prompt_index,
            "prompt_length":     len(prompt_text),
            "has_image":         image_path is not None,
            "rtt_ms":            rtt_ms,
            "response_length":   0,
            "llm_time_ms":       0,
            "status":            f"error: {str(e)}",
            "agreed":            None,
            "rounds":            None,
            "total_exchanges":   None,
            "timestamp":         datetime.now().isoformat(),
        }
        print(f"    ✗ {worker['id']} failed: {e}")
    return result

# ─── Run all patterns under each real edge node condition ─────────────────────
def run_patterns_under_edge_conditions(live_nodes, prompts):
    results = []

    print(f"\n{'='*65}")
    print("RUNNING PATTERNS UNDER REAL EDGE NODE CONDITIONS")
    print(f"{'='*65}")

    for node in live_nodes:
        tc_cmd = build_tc_from_edge_node(node)

        print(f"\n>>> Edge Node : {node['node']} "
              f"({node['provider']}, {node['mean_ms']}ms, "
              f"{node['loss_pct']}% loss)")
        print(f"    TC command : {tc_cmd}")

        apply_edge_condition(tc_cmd)

        for prompt in prompts:
            image_path = prompt.get("image_path", None)

            # ── Coordinator ───────────────────────────────────────────────────
            print(f"\n  [COORDINATOR] {prompt['type'].upper()} "
                  f"#{prompt['index']}"
                  + (f" | image" if image_path else ""))
            for worker in WORKERS:
                # Same guard as original coordinator benchmark
                if prompt["type"] == "image" and worker["model"] != "llava":
                    print(f"    - Skipping {worker['id']} ({worker['model']}) "
                          f"— image prompt")
                    continue
                result = send_to_worker(
                    worker,
                    prompt["text"],
                    image_path,
                    node["node"],
                    prompt["type"],
                    prompt["index"],
                    node,
                    tc_cmd,
                )
                results.append(result)

            # ── Negotiation (text prompts only) ───────────────────────────────
            if prompt["type"] != "image":
                print(f"\n  [NEGOTIATION] {prompt['type'].upper()} "
                      f"#{prompt['index']}")
                try:
                    start  = time.time()
                    result = run_negotiation(
                        topic=prompt["text"],
                        network_condition=node["node"],
                        max_rounds=4,
                    )
                    elapsed = round((time.time() - start) * 1000, 2)
                    results.append({
                        "pattern":           "negotiation",
                        "edge_node":         node["node"],
                        "provider":          node["provider"],
                        "region":            node["region"],
                        "measured_latency":  node["mean_ms"],
                        "measured_jitter":   node["jitter_ms"],
                        "measured_loss_pct": node["loss_pct"],
                        "tc_cmd_applied":    tc_cmd,
                        "worker_id":         None,
                        "model":             None,
                        "prompt_type":       prompt["type"],
                        "prompt_index":      prompt["index"],
                        "prompt_length":     len(prompt["text"]),
                        "has_image":         False,
                        "rtt_ms":            None,
                        "response_length":   None,
                        "llm_time_ms":       None,
                        "status":            "success",
                        "agreed":            result.get("agreed"),
                        "rounds":            result.get("rounds"),
                        "total_exchanges":   None,
                        "timestamp":         datetime.now().isoformat(),
                    })
                    print(f"  → {elapsed}ms | "
                          f"Agreed: {result.get('agreed')} | "
                          f"Rounds: {result.get('rounds')}")
                except Exception as e:
                    print(f"  → FAILED: {e}")

            # ── P2P ───────────────────────────────────────────────────────────
            print(f"\n  [P2P] {prompt['type'].upper()} #{prompt['index']}")
            try:
                start   = time.time()
                history = run_p2p_exchange(
                    topic=prompt["text"],
                    my_id="peer_agent_1",
                    peer_id="peer_agent_2",
                    network_condition=node["node"],
                    max_rounds=3,
                    image_path=image_path,
                )
                elapsed = round((time.time() - start) * 1000, 2)
                results.append({
                    "pattern":           "p2p",
                    "edge_node":         node["node"],
                    "provider":          node["provider"],
                    "region":            node["region"],
                    "measured_latency":  node["mean_ms"],
                    "measured_jitter":   node["jitter_ms"],
                    "measured_loss_pct": node["loss_pct"],
                    "tc_cmd_applied":    tc_cmd,
                    "worker_id":         None,
                    "model":             None,
                    "prompt_type":       prompt["type"],
                    "prompt_index":      prompt["index"],
                    "prompt_length":     len(prompt["text"]),
                    "has_image":         image_path is not None,
                    "rtt_ms":            None,
                    "response_length":   None,
                    "llm_time_ms":       None,
                    "status":            "success",
                    "agreed":            None,
                    "rounds":            None,
                    "total_exchanges":   len(history),
                    "timestamp":         datetime.now().isoformat(),
                })
                print(f"  → {elapsed}ms | Exchanges: {len(history)}")
            except Exception as e:
                print(f"  → FAILED: {e}")

        reset_network()

    return results

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Step 1: capture live conditions from all edge nodes right now
    live_nodes = capture_all_edge_nodes()

    if not live_nodes:
        print("No edge nodes reachable. Exiting.")
        sys.exit(1)

    # Step 2: all prompt types including image
    prompts = (
        [{"type": "simple",  "index": i, "text": p,
          "image_path": None}
         for i, p in enumerate(SIMPLE_PROMPTS[:2])] +
        [{"type": "complex", "index": i, "text": p,
          "image_path": None}
         for i, p in enumerate(COMPLEX_PROMPTS[:2])] +
        [{"type": "image",   "index": i, "text": p["text"],
          "image_path": p["image_path"]}
         for i, p in enumerate(IMAGE_PROMPTS)]
    )

    # Step 3: run all patterns under real edge conditions
    results = run_patterns_under_edge_conditions(live_nodes, prompts)

    # ─── Save ─────────────────────────────────────────────────────────────────
    out_dir    = "/home/jovyan/multi_agent_benchmark/benchmarks/results"
    os.makedirs(out_dir, exist_ok=True)

    json_path  = f"{out_dir}/edge_node_benchmark.json"
    csv_path   = f"{out_dir}/edge_node_benchmark.csv"
    nodes_path = f"{out_dir}/live_edge_node_measurements.json"

    with open(nodes_path, "w") as f:
        json.dump(live_nodes, f, indent=2)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    if results:
        # Collect ALL fieldnames across all rows — no mismatch error
        all_fields = []
        for row in results:
            for key in row.keys():
                if key not in all_fields:
                    all_fields.append(key)

        # Fill missing fields with None
        for row in results:
            for field in all_fields:
                if field not in row:
                    row[field] = None

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            writer.writerows(results)

    print(f"\n{'='*65}")
    print(f"Done! {len(results)} total runs.")
    print(f"Edge node measurements → {nodes_path}")
    print(f"Benchmark results      → {json_path}")
    print(f"CSV                    → {csv_path}")