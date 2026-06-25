import requests
import base64
import time

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

def call_llm(prompt, model="mistral", stream=False, image_path=None):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }

    # Add image if provided (only llava supports this)
    if image_path:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        payload["images"] = [image_b64]

    start = time.time()
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    llm_time = round((time.time() - start) * 1000, 2)

    result = response.json()
    return result.get("response", ""), llm_time