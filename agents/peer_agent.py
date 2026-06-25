import requests
import time
import sys
sys.path.insert(0, '/home/jovyan/multi_agent_benchmark/core')
from logger import log_message
from llm import call_llm

PEER_MODEL_MAP = {
    "peer_agent_1": "mistral",
    "peer_agent_2": "mistral",
    "peer_agent_3": "llava",
}

PEER_PORTS = {
    "peer_agent_1": 8001,
    "peer_agent_2": 8002,
    "peer_agent_3": 8003,
}

def get_peer_url(peer_id):
    return f"http://127.0.0.1:{PEER_PORTS[peer_id]}"

def run_p2p_exchange(topic, my_id, peer_id, network_condition, max_rounds=3, image_path=None):
    model = PEER_MODEL_MAP[my_id]
    conversation_history = []

    prompt = f"You are edge computing agent {my_id}. Give a brief opening statement (2-3 sentences) about: {topic}"
    my_response, llm_time = call_llm(prompt, model=model)
    log_message(sender=my_id, receiver=peer_id, pattern="peer_to_peer",
                content=my_response, network_condition=network_condition,
                round_number=0, llm_response_time_ms=llm_time)
    conversation_history.append({"role": my_id, "content": my_response})
    print(f"[{my_id} | {model}] Round 0: {my_response[:80]}...")

    peer_url = get_peer_url(peer_id)

    for round_num in range(1, max_rounds + 1):
        payload = {
            "sender": my_id,
            "receiver": peer_id,
            "content": my_response,
            "history": conversation_history,
            "round_number": round_num,
            "pattern": "peer_to_peer",
            "network_condition": network_condition
        }
        start = time.time()
        try:
            r = requests.post(f"{peer_url}/p2p_respond", json=payload, timeout=120)
            rtt_ms = round((time.time() - start) * 1000, 2)
            peer_response = r.json().get("response", "")
            log_message(sender=peer_id, receiver=my_id, pattern="peer_to_peer",
                        content=peer_response, network_condition=network_condition,
                        round_number=round_num, llm_response_time_ms=rtt_ms)
            conversation_history.append({"role": peer_id, "content": peer_response})
            print(f"[{my_id}] Round {round_num} peer responded in {rtt_ms}ms: {peer_response[:80]}...")

            history_text = "\n".join([f"{h['role']}: {h['content']}" for h in conversation_history[-4:]])
            prompt = f"You are {my_id}. Continue this discussion adding a new point:\n{history_text}\n{my_id}:"
            my_response, llm_time = call_llm(prompt, model=model)
            log_message(sender=my_id, receiver=peer_id, pattern="peer_to_peer",
                        content=my_response, network_condition=network_condition,
                        round_number=round_num, llm_response_time_ms=llm_time)
            conversation_history.append({"role": my_id, "content": my_response})

        except Exception as e:
            print(f"[{my_id}] Round {round_num} failed: {e}")
            break

    return conversation_history


if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "edge computing vs cloud computing tradeoffs"
    condition = sys.argv[2] if len(sys.argv) > 2 else "baseline"
    print(f"\n[P2P] Topic: {topic}")
    print(f"[P2P] Network condition: {condition}\n")
    history = run_p2p_exchange(topic=topic, my_id="peer_agent_1", peer_id="peer_agent_2",
                                network_condition=condition, max_rounds=3)
    print(f"\n[P2P] Completed {len(history)} total exchanges")