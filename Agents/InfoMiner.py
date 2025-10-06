import os
import re
import json
import google.generativeai as genai
from datetime import datetime, timedelta

# ---------------------------
# Gemini Setup
# ---------------------------
def setup_gemini(api_key: str, model: str = "gemini-2.5-pro"):
    genai.configure(api_key=api_key)
    return model


# ---------------------------
# Cache Handling
# ---------------------------
CACHE_FILE = "info_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("[Cache Error] Corrupted cache. Resetting...")
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ---------------------------
# Prompt Generator
# ---------------------------
def generate_info_prompt(library: str, model_name: str):
    """Generates library-specific documentation prompt"""
    base_prompt = f"""
You are a machine learning documentation expert. Retrieve detailed documentation for model '{model_name}' from the '{library}' library.

Tasks:
1. Provide a **short description** of what this model does.
2. Extract **all parameters** of the `__init__` method, with type, default value, and short explanation.
3. List **important attributes** (after fitting/training).
4. Return a valid JSON object in the following format:

{{
  "model_name": "{model_name}",
  "library": "{library}",
  "description": "...",
  "init_parameters": {{
     "param1": {{"type": "...", "default": "...", "description": "..."}},
     "param2": {{"type": "...", "default": "...", "description": "..."}}
  }},
  "attributes": {{
     "attr1": {{"type": "...", "description": "..."}},
     "attr2": {{"type": "...", "description": "..."}}
  }}
}}
"""

    # Add official reference URLs
    if library.lower() == "pyod":
        base_prompt += "\nOfficial docs: https://pyod.readthedocs.io/en/latest/pyod.models.html"
    elif library.lower() == "pygod":
        base_prompt += f"\nOfficial docs: https://docs.pygod.org/en/latest/pygod.detector.{model_name}.html"
    elif library.lower() == "tslib":
        base_prompt += f"\nOfficial repo: https://github.com/thuml/Time-Series-Library/"
    else:
        base_prompt += "\nUse general ML documentation if not found."

    return base_prompt.strip()


# ---------------------------
# Info Miner Core
# ---------------------------
def info_miner(selection: dict, model_name: str = "gemini-2.5-pro"):
    """
    Info Miner Agent for AD-AGENT.
    Uses Gemini 2.5 Pro to fetch and cache model documentation.
    """

    package = selection.get("package", "pyod")
    model = selection.get("model", None)
    if not model:
        raise ValueError("No model provided in selection output.")

    cache = load_cache()

    # --- Check cache validity (7 days)
    if model in cache:
        try:
            cached_time = datetime.fromisoformat(cache[model]["timestamp"])
            if datetime.now() - cached_time < timedelta(days=7):
                print(f"[Info Miner] Cache hit for {model}")
                return cache[model]["data"]
            else:
                print(f"[Info Miner] Cache expired for {model}")
        except Exception:
            print("[Info Miner] Cache parse error, regenerating...")

    # --- Query Gemini for documentation ---
    print(f"[Info Miner] Querying Gemini for model: {model} ({package})")
    gen_model = genai.GenerativeModel(model_name)

    prompt = generate_info_prompt(package, model)
    response = gen_model.generate_content(prompt)
    raw_output = response.text.strip()

    # Extract JSON from response
    json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not json_match:
        print("[Info Miner] Failed to extract JSON; returning raw text")
        data = {"raw_text": raw_output}
    else:
        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            print("[Info Miner] Invalid JSON structure; returning raw text")
            data = {"raw_text": raw_output}

    # --- Save to cache ---
    cache[model] = {
        "timestamp": datetime.now().isoformat(),
        "data": data
    }
    save_cache(cache)
    print(f"[Info Miner] Cached documentation for {model}")

    return data

