import requests
import time
import sys
sys.path.insert(0, '/home/jovyan/multi_agent_benchmark/core')
from logger import log_message
from llm import call_llm

# Map each agent to its own model
AGENT_MODEL_MAP = {
    "agent_1": "mistral",
    "agent_2": "mistral",
    "agent_3": "llava",
}

AGENT_PORTS = {
    "agent_1": 8001,
    "agent_2": 8002,
    "agent_3": 8003,
}
def get_agent_url(agent_id):
    return f"http://127.0.0.1:{AGENT_PORTS[agent_id]}"

def agent_propose(agent_id, prompt, network_condition, round_number, image_path=None):
    """An agent generates a proposal using its assigned model."""
    model = AGENT_MODEL_MAP[agent_id]
    # Only llava gets the image, all others just get text
    img = image_path if model == "llava" else None
    response, llm_time = call_llm(prompt, model=model, image_path=img)
    log_message(
        sender=agent_id,
        receiver="broadcast",
        pattern="negotiation",
        content=response,
        network_condition=network_condition,
        round_number=round_number,
        llm_response_time_ms=llm_time
    )
    print(f"[{agent_id} | {model}] Round {round_number}: {response[:80]}...")
    return response, llm_time

def run_negotiation(topic, network_condition="baseline", max_rounds=4, image_path=None):
    """
    Multi-agent negotiation:
    - agent_1 acts as the initiator/proposer
    - agent_2 to agent_6 each respond in turn per round
    - If any agent signals agreement, negotiation ends early
    - image_path is optional, only used by llava (agent_6)
    """
    print(f"\n[Negotiation] Topic: {topic}")
    print(f"[Negotiation] Network condition: {network_condition}")
    print(f"[Negotiation] Agents: {list(AGENT_MODEL_MAP.keys())}")
    if image_path:
        print(f"[Negotiation] Image provided (llava will use it): {image_path}\n")

    agents = list(AGENT_MODEL_MAP.keys())
    initiator = agents[0]       # agent_1 starts
    responders = agents[1:]     # agent_2 to agent_6 respond

    # agent_1 makes initial proposal
    prompt = (
        f"You are negotiation {initiator}. Make a specific initial proposal about: {topic}. "
        f"Be concise (2-3 sentences)."
    )
    current_proposal, _ = agent_propose(
        initiator, prompt, network_condition, round_number=0, image_path=image_path
    )

    for round_num in range(1, max_rounds + 1):
        print(f"\n--- Round {round_num} ---")
        round_responses = []

        for responder in responders:
            model = AGENT_MODEL_MAP[responder]
            payload = {
                "sender": initiator,
                "receiver": responder,
                "content": current_proposal,
                "round_number": round_num,
                "pattern": "negotiation",
                "network_condition": network_condition
            }

            start = time.time()
            try:
                url = get_agent_url(responder)
                r = requests.post(f"{url}/negotiate", json=payload, timeout=120)
                rtt_ms = round((time.time() - start) * 1000, 2)
                data = r.json()

                counter = data.get("counter_proposal", "")
                agreed = data.get("agreed", False)

                log_message(
                    sender=responder,
                    receiver=initiator,
                    pattern="negotiation",
                    content=counter,
                    network_condition=network_condition,
                    round_number=round_num,
                    llm_response_time_ms=rtt_ms
                )
                print(f"  [{responder} | {model}] RTT {rtt_ms}ms: {counter[:80]}...")
                round_responses.append({
                    "agent": responder,
                    "model": model,
                    "counter": counter,
                    "agreed": agreed,
                    "rtt_ms": rtt_ms
                })

                if agreed:
                    print(f"\n[Negotiation] AGREEMENT reached by {responder} at round {round_num}!")
                    log_message(
                        sender="system", receiver="system",
                        pattern="negotiation",
                        content=f"AGREEMENT by {responder} at round {round_num}",
                        network_condition=network_condition,
                        round_number=round_num
                    )
                    return {
                        "rounds": round_num,
                        "agreed": True,
                        "agreed_by": responder,
                        "final": counter,
                        "responses": round_responses
                    }

            except Exception as e:
                print(f"  [{responder}] Round {round_num} failed: {e}")

        # agent_1 refines proposal based on all responses this round
        all_counters = " | ".join([r["counter"][:60] for r in round_responses if r["counter"]])
        prompt = (
            f"You are {initiator} negotiating about {topic}. "
            f"The other agents responded: {all_counters}. "
            f"Refine your proposal considering their feedback (2-3 sentences)."
        )
        current_proposal, _ = agent_propose(
            initiator, prompt, network_condition, round_number=round_num, image_path=image_path
        )

    print(f"\n[Negotiation] Max rounds reached. No full agreement.")
    return {
        "rounds": max_rounds,
        "agreed": False,
        "final": current_proposal
    }


if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "optimal bandwidth allocation for edge nodes"
    condition = sys.argv[2] if len(sys.argv) > 2 else "baseline"
    image = sys.argv[3] if len(sys.argv) > 3 else None

    result = run_negotiation(topic=topic, network_condition=condition, max_rounds=4, image_path=image)
    print(f"\n[Negotiation] Final Result: {result}")