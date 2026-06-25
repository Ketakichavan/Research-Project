import sys
import json
import csv
import os
import time
from datetime import datetime

sys.path.insert(0, '/home/jovyan/multi_agent_benchmark')
sys.path.insert(0, '/home/jovyan/multi_agent_benchmark/core')

from agents.peer_agent import run_p2p_exchange
from benchmarks.prompts import SIMPLE_PROMPTS, COMPLEX_PROMPTS, IMAGE_PROMPTS

NETWORK_CONDITIONS = [
    {"name": "baseline",             "cmd": None},
    {"name": "100ms_delay",          "cmd": "sudo tc qdisc add dev lo root netem delay 100ms"},
    {"name": "jitter_50ms_gaussian", "cmd": "sudo tc qdisc add dev lo root netem delay 100ms 50ms distribution normal"},
    {"name": "packet_loss_5percent", "cmd": "sudo tc qdisc add dev lo root netem loss 5%"},
    {"name": "iot_edge_extreme",     "cmd": "sudo tc qdisc add dev lo root netem delay 500ms loss 2% rate 256kbit"}
]

RESET_CMD = "sudo tc qdisc del dev lo root 2>/dev/null; true"

# All peer pairs for 3 agents
PEER_PAIRS = [
    ("peer_agent_1", "peer_agent_2"),
    ("peer_agent_2", "peer_agent_3"),
    ("peer_agent_1", "peer_agent_3"),
]


def apply_network(condition):
    os.system(RESET_CMD)
    if condition["cmd"]:
        os.system(condition["cmd"])
        time.sleep(1)
    print(f"\n>>> Network condition: {condition['name']}")


def reset_network():
    os.system(RESET_CMD)


def run_p2p_benchmark():
    results = []

    # Combine all prompts and standardise the dictionary structure
    all_prompts = (
        [{"type": "simple",  "index": i, "text": p,         "image_path": None} for i, p in enumerate(SIMPLE_PROMPTS)] +
        [{"type": "complex", "index": i, "text": p,         "image_path": None} for i, p in enumerate(COMPLEX_PROMPTS)] +
        [{"type": "image",   "index": i, "text": p["text"], "image_path": p["image_path"]} for i, p in enumerate(IMAGE_PROMPTS)]
    )

    total = len(NETWORK_CONDITIONS) * len(all_prompts) * len(PEER_PAIRS)
    count = 0

    for condition in NETWORK_CONDITIONS:
        apply_network(condition)

        print(f"\n{'='*60}")
        print(f"Network Condition: {condition['name']}")
        print(f"{'='*60}")

        for prompt in all_prompts:
            for my_id, peer_id in PEER_PAIRS:
                count += 1
                print(f"\n[{count}/{total}] {prompt['type'].upper()} prompt #{prompt['index']}")
                print(f"Pair: {my_id} <-> {peer_id}")
                print(f"Topic: {prompt['text'][:80]}...")

                try:
                    start = time.time()
                    history = run_p2p_exchange(
                        topic=prompt["text"],
                        my_id=my_id,
                        peer_id=peer_id,
                        network_condition=condition["name"],
                        max_rounds=3,
                        image_path=prompt["image_path"]
                    )
                    total_time = round((time.time() - start) * 1000, 2)

                    row = {
                        "pattern": "peer_to_peer",
                        "network_condition": condition["name"],
                        "prompt_type": prompt["type"],
                        "prompt_index": prompt["index"],
                        "prompt_length": len(prompt["text"]),
                        "has_image": prompt["image_path"] is not None,
                        "my_id": my_id,
                        "peer_id": peer_id,
                        "total_exchanges": len(history),
                        "total_time_ms": total_time,
                    }
                    results.append(row)
                    print(f"  → Exchanges: {len(history)} | Time: {total_time}ms")

                except Exception as e:
                    print(f"  → FAILED: {e}")
                    results.append({
                        "pattern": "peer_to_peer",
                        "network_condition": condition["name"],
                        "prompt_type": prompt["type"],
                        "prompt_index": prompt["index"],
                        "prompt_length": len(prompt["text"]),
                        "has_image": prompt["image_path"] is not None,
                        "my_id": my_id,
                        "peer_id": peer_id,
                        "total_exchanges": None,
                        "total_time_ms": None,
                    })

        reset_network()

    # Save results
    out_dir = "/home/jovyan/multi_agent_benchmark/benchmarks/results"
    os.makedirs(out_dir, exist_ok=True)

    json_path = f"{out_dir}/p2p_results_benchmark.json"
    csv_path  = f"{out_dir}/p2p_results_benchmark.csv"

    existing = []
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            existing = json.load(f)

    existing.extend(results)
    with open(json_path, "w") as f:
        json.dump(existing, f, indent=2)

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"P2P Benchmark complete! {len(results)} total runs.")
    print(f"JSON → {json_path}")
    print(f"CSV  → {csv_path}")


if __name__ == "__main__":
    run_p2p_benchmark()