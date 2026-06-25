import requests
import time
import sys
sys.path.insert(0, '/home/jovyan/multi_agent_benchmark/core')
from logger import log_message
from llm import call_llm

# ─── 3 WORKERS CONFIGURED ────────────────────────────────────────────────────
WORKERS = [
    "http://127.0.0.1:8001",
    "http://127.0.0.1:8002",
    "http://127.0.0.1:8003"
]

def send_task(task: str, network_condition: str = "baseline"):
    """Coordinator sends task to all workers and collects results."""
    results = []

    # Step 1: LLM breaks task into 3 subtasks (Updated for 3 workers)
    prompt = f"Break this task into 3 clear subtasks (one line each, numbered):\nTask: {task}"
    llm_response, llm_time = call_llm(prompt)
    log_message(
        sender="coordinator",
        receiver="llm",
        pattern="coordinator_worker",
        content=prompt,
        network_condition=network_condition,
        round_number=0,
        llm_response_time_ms=llm_time
    )

    # Slice updated to 3
    subtasks = llm_response.strip().split("\n")[:3]

    # Step 2: Send each subtask to a worker
    for i, (worker_url, subtask) in enumerate(zip(WORKERS, subtasks)):
        payload = {
            "sender": "coordinator",
            "receiver": f"worker_{i+1}",
            "content": subtask.strip(),
            "round_number": 1,
            "pattern": "coordinator_worker",
            "network_condition": network_condition
        }
        start = time.time()
        try:
            r = requests.post(f"{worker_url}/process", json=payload, timeout=120)
            rtt_ms = round((time.time() - start) * 1000, 2)
            response_data = r.json()
            log_message(
                sender="coordinator",
                receiver=f"worker_{i+1}",
                pattern="coordinator_worker",
                content=subtask.strip(),
                network_condition=network_condition,
                round_number=1,
                llm_response_time_ms=rtt_ms
            )
            results.append({
                "worker": f"worker_{i+1}",
                "subtask": subtask.strip(),
                "response": response_data.get("result", ""),
                "rtt_ms": rtt_ms,
                "model": response_data.get("model", "unknown")
            })
            print(f"[Coordinator] Worker {i+1} responded in {rtt_ms}ms")
        except Exception as e:
            print(f"[Coordinator] Worker {i+1} failed: {e}")

    return results

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "Summarize the benefits of edge computing"
    condition = sys.argv[2] if len(sys.argv) > 2 else "baseline"
    print(f"\n[Coordinator] Starting task: {task}")
    print(f"[Coordinator] Network condition: {condition}\n")
    results = send_task(task, condition)
    print("\n[Coordinator] All results:")
    for r in results:
        print(f"  {r['worker']} ({r['model']}): {r['response'][:100]}...")