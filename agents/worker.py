import sys
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
sys.path.insert(0, '/home/jovyan/multi_agent_benchmark/core')
from logger import log_message
from llm import call_llm

app = FastAPI()

WORKER_ID = "worker_1"
MODEL = "mistral"

MODEL_MAP = {
    "worker_1": "mistral",
    "worker_2": "mistral",
    "worker_3": "llava",
}

class Message(BaseModel):
    sender: str
    receiver: str
    content: str
    round_number: int = 1
    pattern: str = "coordinator_worker"
    network_condition: str = "baseline"

class P2PMessage(BaseModel):
    sender: str
    receiver: str
    content: str
    history: list = []
    round_number: int = 1
    pattern: str = "peer_to_peer"
    network_condition: str = "baseline"

@app.post("/process")
def process(msg: Message):
    print(f"[{WORKER_ID}] Received: {msg.content[:60]}")
    prompt = f"You are an edge computing agent using {MODEL}. Complete this subtask concisely:\n{msg.content}"
    response, llm_time = call_llm(prompt, model=MODEL)
    log_message(sender=WORKER_ID, receiver=msg.sender, pattern=msg.pattern,
                content=response, network_condition=msg.network_condition,
                round_number=msg.round_number, llm_response_time_ms=llm_time)
    print(f"[{WORKER_ID}] Responded in {llm_time}ms")
    return {"worker": WORKER_ID, "result": response, "llm_time_ms": llm_time, "model": MODEL}

@app.post("/p2p_respond")
def p2p_respond(msg: P2PMessage):
    history_text = "\n".join([f"{h['role']}: {h['content']}" for h in msg.history[-4:]])
    prompt = f"You are edge agent {msg.receiver}. Respond with a new insight:\n{history_text}\n{msg.receiver}:"
    response, llm_time = call_llm(prompt, model=MODEL)
    log_message(sender=msg.receiver, receiver=msg.sender, pattern="peer_to_peer",
                content=response, network_condition=msg.network_condition,
                round_number=msg.round_number, llm_response_time_ms=llm_time)
    print(f"[P2P {WORKER_ID}] Round {msg.round_number} responded in {llm_time}ms")
    return {"response": response, "llm_time_ms": llm_time, "model": MODEL}

@app.post("/negotiate")
def negotiate(msg: dict):
    content = msg.get("content", "")
    round_num = msg.get("round_number", 1)
    condition = msg.get("network_condition", "baseline")
    prompt = f"""You are negotiation agent B using {MODEL}. Round {round_num}.
The other agent proposed: {content}
If acceptable, start with AGREED: and summarize the deal.
Otherwise start with COUNTER: and give a counter-proposal (2 sentences)."""
    response, llm_time = call_llm(prompt, model=MODEL)
    agreed = response.strip().startswith("AGREED")
    log_message(sender="agent_B", receiver="agent_A", pattern="negotiation",
                content=response, network_condition=condition,
                round_number=round_num, llm_response_time_ms=llm_time)
    print(f"[Negotiation {WORKER_ID}] Round {round_num}: agreed={agreed}, time={llm_time}ms")
    return {"counter_proposal": response, "agreed": agreed, "llm_time_ms": llm_time, "model": MODEL}

if __name__ == "__main__":
    WORKER_ID = sys.argv[1] if len(sys.argv) > 1 else "worker_1"
    PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8001
    MODEL = MODEL_MAP.get(WORKER_ID, "mistral")
    print(f"[{WORKER_ID}] Starting on port {PORT} using model: {MODEL}")
    uvicorn.run(app, host="127.0.0.1", port=PORT)