import requests
import time
import json
import os
import sys
import csv

sys.path.insert(0, '/home/jovyan/multi_agent_benchmark/core')
from logger import log_message
# 1. Imported all three prompt types
from prompts import SIMPLE_PROMPTS, COMPLEX_PROMPTS, IMAGE_PROMPTS

# ─── ISOLATED NETWORK CONDITIONS ─────────────────────────────────────────────
NETWORK_CONDITIONS = [
    {"name": "baseline",             "cmd": None},
    {"name": "100ms_delay",          "cmd": "sudo tc qdisc add dev lo root netem delay 100ms"},
    {"name": "jitter_50ms_gaussian", "cmd": "sudo tc qdisc add dev lo root netem delay 100ms 50ms distribution normal"},
    {"name": "packet_loss_5percent", "cmd": "sudo tc qdisc add dev lo root netem loss 5%"},
    {"name": "iot_edge_extreme",     "cmd": "sudo tc qdisc add dev lo root netem delay 500ms loss 2% rate 256kbit"}
]

RESET_CMD = "sudo tc qdisc del dev lo root 2>/dev/null; true"

# ─── HETEROGENEOUS WORKERS (Mistral + Llava) ─────────────────────────────────
WORKERS = [
    {"id": "worker_1",  "url": "http://127.0.0.1:8001", "model": "mistral"},
    {"id": "worker_2",  "url": "http://127.0.0.1:8002", "model": "mistral"},
    {"id": "worker_3",  "url": "http://127.0.0.1:8003", "model": "llava"}
]

# ─── Results storage ─────────────────────────────────────────────────────────
results = []
os.makedirs("/home/jovyan/multi_agent_benchmark/benchmarks/results", exist_ok=True)

# ─── Helper Functions ────────────────────────────────────────────────────────
def apply_network(condition):
    os.system(RESET_CMD)
    if condition["cmd"]:
        os.system(condition["cmd"])
        time.sleep(1)
    print(f"\n>>> Network condition: {condition['name']}")

def reset_network():
    os.system(RESET_CMD)

def send_to_worker(worker, prompt_text, image_path, condition_name, prompt_type, prompt_index):
    payload = {
        "sender": "benchmark_runner",
        "receiver": worker["id"],
        "content": prompt_text,
        "round_number": 1,
        "pattern": "coordinator_worker",
        "network_condition": condition_name
    }
    
    # If there is an image, attach it to the payload
    if image_path:
        payload["image_path"] = image_path

    start = time.time()
    try:
        r = requests.post(f"{worker['url']}/process", json=payload, timeout=180)
        rtt_ms = round((time.time() - start) * 1000, 2)
        data = r.json()
        result = {
            "worker_id": worker["id"],
            "model": worker["model"],
            "prompt_type": prompt_type,
            "prompt_index": prompt_index,
            "prompt_length": len(prompt_text),
            "network_condition": condition_name,
            "rtt_ms": rtt_ms,
            "response_length": len(data.get("result", "")),
            "llm_time_ms": data.get("llm_time_ms", 0),
            "status": "success"
        }
        print(f"  ✓ {worker['id']} ({worker['model']}) — {rtt_ms}ms")
    except Exception as e:
        rtt_ms = round((time.time() - start) * 1000, 2)
        result = {
            "worker_id": worker["id"],
            "model": worker["model"],
            "prompt_type": prompt_type,
            "prompt_index": prompt_index,
            "prompt_length": len(prompt_text),
            "network_condition": condition_name,
            "rtt_ms": rtt_ms,
            "response_length": 0,
            "llm_time_ms": 0,
            "status": f"error: {str(e)}"
        }
        print(f"  ✗ {worker['id']} failed: {e}")
    return result

# ─── Main benchmark loop ─────────────────────────────────────────────────────
def run_benchmark():
    all_prompts = (
        [{"type": "simple",  "index": i, "text": p,          "image_path": None} for i, p in enumerate(SIMPLE_PROMPTS)] +
        [{"type": "complex", "index": i, "text": p,          "image_path": None} for i, p in enumerate(COMPLEX_PROMPTS)] +
        [{"type": "image",   "index": i, "text": p["text"],  "image_path": p["image_path"]} for i, p in enumerate(IMAGE_PROMPTS)]
    )

    print(f"Starting ADVANCED ISOLATED benchmark...\n")

    for condition in NETWORK_CONDITIONS:
        apply_network(condition)

        for prompt in all_prompts:
            print(f"\n  [{prompt['type'].upper()} {prompt['index']+1}] {prompt['text'][:60]}...")
            
            for worker in WORKERS:
                # THE GUARD: Stop Mistral from hallucinating on images
                if prompt["type"] == "image" and worker["model"] != "llava":
                    print(f"  - Skipping {worker['id']} ({worker['model']}) to prevent text-model hallucination.")
                    continue
                
                result = send_to_worker(worker, prompt["text"], prompt["image_path"], condition["name"], prompt["type"], prompt["index"])
                results.append(result)

        reset_network()

    # ─── Save Results (Static Filenames) ──────────────────────────────────────
    json_path = "/home/jovyan/multi_agent_benchmark/benchmarks/results/coordinator_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = "/home/jovyan/multi_agent_benchmark/benchmarks/results/coordinator_benchmark.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"\n Benchmark complete!")
    print(f"   JSON: {json_path}")
    print(f"   CSV:  {csv_path}")
    print(f"   Total records: {len(results)}")

if __name__ == "__main__":
    run_benchmark()