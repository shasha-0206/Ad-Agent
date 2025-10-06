import json
import re
import google.generativeai as genai

from Prompts import (
    generate_model_selection_prompt_from_pygod,
    generate_model_selection_prompt_from_pyod,
    generate_model_selection_prompt_from_timeseries,
)

def select_agent(session, model="gemini-2.5-pro"):
    """
    Selector Agent for AD-AGENT (Gemini 2.5 Pro version).
    - Chooses library & model based on Processor output.
    - Uses predefined prompt generators (from Prompts.py).
    - Calls Gemini to recommend the best model.
    - Returns {"package", "model", "reason"}.
    """

    # Handle both dict and session-like objects
    if isinstance(session, dict):
        info = session
    else:
        info = session.read("processor")

    # Extract key fields from processor output
    modality = info.get("inferred_modality", "tabular")
    datasets = info.get("datasets", {})

    # Pick the first dataset if multiple
    dataset_name, dataset_meta = next(iter(datasets.items()), ("unknown", {}))
    shape = dataset_meta.get("shape", [0, 0])

    # Step 1: Select package & build appropriate prompt
    if modality in ("tabular", "univariate"):
        package = "pyod"
        prompt = generate_model_selection_prompt_from_pyod(
            name=dataset_name,
            size=shape[0],
            dim=shape[1] if len(shape) > 1 else 1,
        )

    elif modality == "graph":
        package = "pygod"
        prompt = generate_model_selection_prompt_from_pygod(
            name=dataset_name,
            num_node=dataset_meta.get("graph_n_nodes", 100),
            num_edge=dataset_meta.get("graph_n_edges", 200),
            num_feature=shape[1] if len(shape) > 1 else 1,
            avg_degree=dataset_meta.get("avg_degree", 2.0),
        )

    elif modality == "timeseries":
        package = "tslib"
        prompt = generate_model_selection_prompt_from_timeseries(
            name=dataset_name,
            size=shape[0],
            dim=shape[1] if len(shape) > 1 else 1,
            type=dataset_meta.get("type", "univariate"),
        )

    else:
        # Fallback to tabular (PyOD)
        package = "pyod"
        prompt = generate_model_selection_prompt_from_pyod(
            name=dataset_name,
            size=shape[0],
            dim=shape[1] if len(shape) > 1 else 1,
        )

    # Step 2: Combine system + user messages into a single string for Gemini
    prompt_text = ""
    for msg in prompt:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        prompt_text += f"[{role}]\n{content}\n\n"

    # Step 3: Call Gemini API
    model_client = genai.GenerativeModel(model)
    response = model_client.generate_content(prompt_text)
    raw_content = response.text.strip()

    # Step 4: JSON-safe parsing
    json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
        except json.JSONDecodeError:
            parsed = {"choice": "default", "reason": "Failed to parse LLM JSON"}
    else:
        parsed = {"choice": "default", "reason": "Failed to parse LLM JSON"}

    # Step 5: Save & return selection
    selection = {
        "package": package,
        "model": parsed.get("choice"),
        "reason": parsed.get("reason"),
    }

    # Store in dict if possible, else call .write()
    if isinstance(session, dict):
        session["selection"] = selection
    else:
        session.write("selection", selection)

    return selection
