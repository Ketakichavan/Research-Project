import json
import os
from datetime import datetime

LOG_FILE = os.path.expanduser("~/multi_agent_benchmark/benchmarks/results/agent_messages.jsonl")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log_message(sender, receiver, pattern, content,
                network_condition, round_number=None,
                llm_response_time_ms=None):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "sender": sender,
        "receiver": receiver,
        "pattern": pattern,
        "message_size_bytes": len(content.encode("utf-8")),
        "round_number": round_number,
        "network_condition": network_condition,
        "llm_response_time_ms": llm_response_time_ms,
        "content_preview": content[:100]
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record