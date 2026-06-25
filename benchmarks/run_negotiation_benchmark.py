import sys
import json
import csv
import os
import time
from datetime import datetime

sys.path.insert(0, '/home/jovyan/multi_agent_benchmark')
sys.path.insert(0, '/home/jovyan/multi_agent_benchmark/core')

from agents.negotiation_agent import run_negotiation
from benchmarks.prompts import SIMPLE_PROMPTS, COMPLEX_PROMPTS, IMAGE_PROMPTS

NETWORK_CONDITIONS = [
    {"name": "baseline",             "cmd": None},
    {"name": "100ms_delay",          "cmd": "sudo tc qdisc add dev lo root netem delay 100ms"},
    {"name": "jitter_50ms_gaussian", "cmd": "sudo tc qdisc add dev lo root netem delay 100ms 50ms distribution normal"},
    {"name": "packet_loss_5percent", "cmd": "sudo tc qdisc add dev lo root netem loss 5%"},
    {"name": "iot_edge_extreme",     "cmd": "sudo tc qdisc add dev lo root netem delay 500ms loss 2% rate 256kbit"}
]

RESET_CMD = "sudo tc qdisc del dev lo root 2>/dev/null; true"


def apply_network(condition):
    os.system(RESET_CMD)
    if condition["cmd"]:
        os.system(condition["cmd"])
        time.sleep(1)
    print(f"\n>>> Network condition: {condition['name']}")


def reset_network():
    os.system(RESET_CMD)


def generate_sample_images():
    """Generate sample images for llava testing if they don't exist."""
    import matplotlib.pyplot as plt
    import numpy as np

    img_dir = "/home/jovyan/multi_agent_benchmark/benchmarks/images"
    os.makedirs(img_dir, exist_ok=True)

    if not os.path.exists(f"{img_dir}/network_topology.png"):
        fig, ax = plt.subplots(figsize=(8, 6))
        nodes = {
            'Edge Node 1': (1, 3), 'Edge Node 2': (3, 3),
            'Cloud':        (2, 5), 'IoT Device 1': (0, 1),
            'IoT Device 2': (2, 1), 'IoT Device 3': (4, 1)
        }
        for name, (x, y) in nodes.items():
            ax.plot(x, y, 'bo', markersize=15)
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
        edges = [
            ('IoT Device 1', 'Edge Node 1'), ('IoT Device 2', 'Edge Node 1'),
            ('IoT Device 2', 'Edge Node 2'), ('IoT Device 3', 'Edge Node 2'),
            ('Edge Node 1', 'Cloud'),         ('Edge Node 2', 'Cloud')
        ]
        for a, b in edges:
            x1, y1 = nodes[a]; x2, y2 = nodes[b]
            ax.plot([x1, x2], [y1, y2], 'k-')
        ax.set_title('Network Topology Diagram')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{img_dir}/network_topology.png")
        plt.close()
        print(f"  Created: network_topology.png")

    if not os.path.exists(f"{img_dir}/system_arch.png"):
        fig, ax = plt.subplots(figsize=(8, 6))
        layers = ['IoT Sensors', 'Edge Layer', 'Fog Layer', 'Cloud Layer']
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        for i, (layer, color) in enumerate(zip(layers, colors)):
            rect = plt.Rectangle((0.1, i * 0.22 + 0.05), 0.8, 0.18,
                                  color=color, ec='black')
            ax.add_patch(rect)
            ax.text(0.5, i * 0.22 + 0.14, layer,
                    ha='center', va='center', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Edge Computing System Architecture')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{img_dir}/system_arch.png")
        plt.close()
        print(f"  Created: system_arch.png")

    if not os.path.exists(f"{img_dir}/performance_graph.png"):
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        ax.plot(x, np.sin(x) * 10 + 50,  label='Edge Node 1 Latency (ms)')
        ax.plot(x, np.cos(x) * 15 + 70,  label='Edge Node 2 Latency (ms)')
        ax.plot(x, np.random.normal(60, 5, 100), alpha=0.5, label='Baseline')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Network Performance Over Time')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{img_dir}/performance_graph.png")
        plt.close()
        print(f"  Created: performance_graph.png")


def run_negotiation_benchmark():
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Checking/generating sample images for llava...")
    generate_sample_images()

    all_prompts = (
        [{"type": "simple",  "index": i, "text": p,         "image_path": None}
         for i, p in enumerate(SIMPLE_PROMPTS)] +
        [{"type": "complex", "index": i, "text": p,         "image_path": None}
         for i, p in enumerate(COMPLEX_PROMPTS)] +
        [{"type": "image",   "index": i, "text": p["text"], "image_path": p["image_path"]}
         for i, p in enumerate(IMAGE_PROMPTS)]
    )

    total = len(NETWORK_CONDITIONS) * len(all_prompts)
    count = 0

    for condition in NETWORK_CONDITIONS:
        apply_network(condition)

        print(f"\n{'='*60}")
        print(f"Network Condition: {condition['name']}")
        print(f"{'='*60}")

        for prompt in all_prompts:
            count += 1
            print(f"\n[{count}/{total}] {prompt['type'].upper()} prompt #{prompt['index']}")
            print(f"Topic: {prompt['text'][:80]}...")
            if prompt["image_path"]:
                print(f"Image: {os.path.basename(prompt['image_path'])} (llava only)")

            try:
                start = time.time()
                result = run_negotiation(
                    topic=prompt["text"],
                    network_condition=condition["name"],
                    max_rounds=4,
                    image_path=prompt["image_path"]
                )
                total_time = round((time.time() - start) * 1000, 2)

                row = {
                    "timestamp": timestamp,
                    "pattern": "negotiation",
                    "network_condition": condition["name"],
                    "prompt_type": prompt["type"],
                    "prompt_index": prompt["index"],
                    "prompt_length": len(prompt["text"]),
                    "has_image": prompt["image_path"] is not None,
                    "rounds": result.get("rounds"),
                    "agreed": result.get("agreed"),
                    "agreed_by": result.get("agreed_by", "none"),
                    "total_time_ms": total_time,
                }
                results.append(row)
                print(f"  → Agreed: {result.get('agreed')} | Rounds: {result.get('rounds')} | Time: {total_time}ms")

            except Exception as e:
                print(f"  → FAILED: {e}")
                results.append({
                    "timestamp": timestamp,
                    "pattern": "negotiation",
                    "network_condition": condition["name"],
                    "prompt_type": prompt["type"],
                    "prompt_index": prompt["index"],
                    "prompt_length": len(prompt["text"]),
                    "has_image": prompt["image_path"] is not None,
                    "rounds": None,
                    "agreed": None,
                    "agreed_by": "error",
                    "total_time_ms": None,
                })

        reset_network()

    # Save results
    out_dir = "/home/jovyan/multi_agent_benchmark/benchmarks/results"
    os.makedirs(out_dir, exist_ok=True)

    json_path = f"{out_dir}/benchmark_negotiation_results.json"
    csv_path  = f"{out_dir}/benchmark_negotiation_results.csv"

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
    print(f"Benchmark complete! {len(results)} total runs.")
    print(f"JSON → {json_path}")
    print(f"CSV  → {csv_path}")


if __name__ == "__main__":
    run_negotiation_benchmark()