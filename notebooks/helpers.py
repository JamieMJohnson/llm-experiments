"""
Helper functions for running experiments with open-weight LLMs.
Supports local models via Ollama and hosted models via Together AI / HF.
"""

import json
import os
import time
import csv
from datetime import datetime
from pathlib import Path

import requests

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# LOCAL MODELS (Ollama)
# ============================================================

def query_ollama(prompt, model="phi3:mini", system=None, temperature=0.7, max_tokens=512):
    """Query a local model via Ollama's API.

    Args:
        prompt: The user prompt
        model: Ollama model name (e.g. 'phi3:mini', 'llama3.2:1b')
        system: Optional system prompt
        temperature: Sampling temperature
        max_tokens: Max tokens to generate

    Returns:
        dict with 'response', 'model', 'duration_ms', 'tokens_generated'
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if system:
        payload["system"] = system

    start = time.time()
    resp = requests.post(url, json=payload, timeout=300)
    elapsed = (time.time() - start) * 1000

    data = resp.json()
    return {
        "response": data.get("response", ""),
        "model": model,
        "duration_ms": round(elapsed, 1),
        "tokens_generated": data.get("eval_count", 0),
        "provider": "ollama",
    }


def list_ollama_models():
    """List models currently available in Ollama."""
    resp = requests.get("http://localhost:11434/api/tags")
    return [m["name"] for m in resp.json().get("models", [])]


# ============================================================
# HOSTED MODELS (Together AI)
# ============================================================

def query_together(prompt, model="meta-llama/Llama-3-8b-chat-hf", system=None,
                   temperature=0.7, max_tokens=512, api_key=None):
    """Query an open-weight model hosted on Together AI.

    Free tier: ~$1 of credits on signup (enough for thousands of queries).
    Sign up: https://api.together.xyz/signup

    Popular models:
        - meta-llama/Llama-3-8b-chat-hf
        - meta-llama/Llama-3-70b-chat-hf
        - mistralai/Mistral-7B-Instruct-v0.3
        - mistralai/Mixtral-8x7B-Instruct-v0.1
        - Qwen/Qwen2-72B-Instruct
    """
    api_key = api_key or os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("Set TOGETHER_API_KEY env var or pass api_key. Sign up: https://api.together.xyz/signup")

    url = "https://api.together.xyz/v1/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    start = time.time()
    resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}"}, json={
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }, timeout=120)
    elapsed = (time.time() - start) * 1000

    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"Together AI error: {data['error']}")

    choice = data["choices"][0]
    return {
        "response": choice["message"]["content"],
        "model": model,
        "duration_ms": round(elapsed, 1),
        "tokens_generated": data.get("usage", {}).get("completion_tokens", 0),
        "provider": "together",
    }


# ============================================================
# HOSTED MODELS (Hugging Face Inference API)
# ============================================================

def query_hf(prompt, model="mistralai/Mistral-7B-Instruct-v0.3", max_tokens=512, token=None):
    """Query a model via Hugging Face's free Inference API.

    Free tier: rate-limited but no cost.
    Token: https://huggingface.co/settings/tokens
    """
    token = token or os.environ.get("HF_TOKEN")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    start = time.time()
    resp = requests.post(url, headers=headers, json={
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_tokens},
    }, timeout=120)
    elapsed = (time.time() - start) * 1000

    data = resp.json()
    if isinstance(data, list) and len(data) > 0:
        text = data[0].get("generated_text", "")
    elif isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"HF error: {data['error']}")
    else:
        text = str(data)

    return {
        "response": text,
        "model": model,
        "duration_ms": round(elapsed, 1),
        "tokens_generated": None,
        "provider": "huggingface",
    }


# ============================================================
# EXPERIMENT UTILITIES
# ============================================================

def run_experiment(name, prompts, query_fn, **kwargs):
    """Run a batch of prompts through a model and save results.

    Args:
        name: Experiment name (used for output filename)
        prompts: List of prompt strings, or list of dicts with 'prompt' and optional metadata
        query_fn: One of query_ollama, query_together, query_hf
        **kwargs: Passed to query_fn (model, temperature, etc.)

    Returns:
        List of result dicts
    """
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, p in enumerate(prompts):
        if isinstance(p, str):
            prompt_text = p
            metadata = {}
        else:
            prompt_text = p["prompt"]
            metadata = {k: v for k, v in p.items() if k != "prompt"}

        print(f"  [{i+1}/{len(prompts)}] Querying...", end="", flush=True)
        try:
            result = query_fn(prompt_text, **kwargs)
            result["prompt"] = prompt_text
            result.update(metadata)
            result["status"] = "ok"
        except Exception as e:
            result = {
                "prompt": prompt_text,
                "response": "",
                "status": f"error: {e}",
                **metadata,
            }
        results.append(result)
        print(f" done ({result.get('duration_ms', '?')}ms)")

    # Save to CSV
    outfile = RESULTS_DIR / f"{name}_{timestamp}.csv"
    if results:
        keys = list(results[0].keys())
        with open(outfile, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {outfile}")

    # Also save as JSON for richer data
    json_file = RESULTS_DIR / f"{name}_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump({"experiment": name, "timestamp": timestamp, "params": kwargs, "results": results}, f, indent=2)

    return results


def compare_models(prompt, models_and_fns, n=1, **common_kwargs):
    """Run the same prompt across multiple models for comparison.

    Args:
        prompt: Single prompt string
        models_and_fns: List of (model_name, query_fn, extra_kwargs) tuples
        n: Number of times to repeat each query
        **common_kwargs: Passed to all query functions

    Returns:
        List of result dicts with model comparison data
    """
    results = []
    for model_name, query_fn, extra_kwargs in models_and_fns:
        merged = {**common_kwargs, **extra_kwargs}
        for i in range(n):
            print(f"  {model_name} (run {i+1}/{n})...", end="", flush=True)
            try:
                result = query_fn(prompt, **merged)
                result["run"] = i + 1
                result["status"] = "ok"
            except Exception as e:
                result = {"model": model_name, "response": "", "status": f"error: {e}", "run": i + 1}
            result["prompt"] = prompt
            results.append(result)
            print(f" done")
    return results
