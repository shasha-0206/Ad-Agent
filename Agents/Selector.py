import json
import google.generativeai as genai

from Agents.Prompts import (
    generate_model_selection_prompt_from_pygod,
    generate_model_selection_prompt_from_pyod,
    generate_model_selection_prompt_from_timeseries,
)

def select_agent(session, model="gemini-2.5-pro"):
    """
    Functional Selector Agent for AD-AGENT (Gemini 2.5 Pro version).
    - Decides which library to use based on dataset modality.
    - Builds the appropriate model-selection prompt.
    - Calls Gemini to get best model choice.
    - Stores {"package", "model", "reason"} in session.
    """

    info = session.read("processor")
    modality = info.get("modality", "tabular")
    dataset_meta = info.get("dataset", {}).get("meta", {})

    # Step 1: Build prompt based on modality
    if modality in ("tabular", "univariate"):
        package = "pyod"
        prompt = generate_model_selection_prompt_from_pyod(
            name=dataset_meta.get("name", "unknown"),
            size=info["dataset"]["X"].shape[0],
            dim=info["dataset"]["X"].shape[1] if len(info["dataset"]["X"].shape) > 1 else 1,
        )

    elif modality == "graph":
        package = "pygod"
        prompt = generate_model_selection_prompt_from_pygod(
            name=dataset_meta.get("name", "unknown"),
            num_node=dataset_meta.get("num_node", 100),
            num_edge=dataset_meta.get("num_edge", 200),
            num_feature=dataset_meta.get("num_feature", 10),
            avg_degree=dataset_meta.get("avg_degree", 2.0),
        )

    else:  # fallback = time series
        package = "tslib"
        prompt = generate_model_selection_prompt_from_timeseries(
            name=dataset_meta.get("name", "unknown"),
            size=info["dataset"]["X"].shape[0],
            dim=info["dataset"]["X"].shape[1] if len(info["dataset"]["X"].shape) > 1 else 1,
            type=dataset_meta.get("type", "univariate"),
        )

    # Step 2: Call Gemini API
    model_client = genai.GenerativeModel(model)
    response = model_client.generate_content(prompt[0]["content"])  # our prompt list has only user message

    raw_content = response.text.strip()

    # Step 3: Parse JSON
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        parsed = {"reason": "Failed to parse LLM JSON", "choice": "default"}

    # Step 4: Save selection
    selection = {
        "package": package,
        "model": parsed.get("choice"),
        "reason": parsed.get("reason"),
    }
    session.write("selection", selection)

    return selection
