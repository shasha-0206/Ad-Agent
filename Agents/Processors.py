import os
import re
import json
import pandas as pd
import numpy as np
import scipy.io
import networkx as nx
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# Gemini setup
# ---------------------------
def setup_gemini(api_key: str, model: str = "gemini-2.5-pro"):
    genai.configure(api_key=api_key)
    return model


# ---------------------------
# Parsing and constraints
# ---------------------------
CMD_PATTERN = re.compile(
    r'(?P<action>run|train|evaluate|detect|predict)\s+(?P<algo>[\w\-]+)?(?:\s+on\s+|\s+using\s+)?(?P<paths>.+)',
    flags=re.I
)

def parse_command(cmd: str):
    m = CMD_PATTERN.search(cmd.strip())
    out = {"raw": cmd, "action": None, "algorithm": None, "paths": [], "hints": cmd}
    if m:
        out["action"] = m.group("action").lower()
        if m.group("algo"):
            out["algorithm"] = m.group("algo").strip()
        paths_part = m.group("paths").strip()

        # Remove inline constraints like contamination=0.1
        paths_part_clean = re.sub(r'\b\w+\s*=\s*[\d\.]+\b', '', paths_part)

        # Split by spaces, commas, or &
        files = [p for p in re.split(r'[ ,&]+', paths_part_clean) 
         if p and p.lower() not in ["in", "on", "using"]]
        out["paths"] = files
    return out


def extract_constraints(text: str):
    constraints = {}
    m = re.search(r'contamination\s*=\s*([0-9]*\.?[0-9]+)', text, flags=re.I)
    if m:
        constraints["contamination"] = float(m.group(1))

    m = re.search(r'label\s*column\s*[:=]\s*([A-Za-z0-9_]+)', text, flags=re.I)
    if m:
        constraints["label_col"] = m.group(1)

    if re.search(r'\bsupervis(ed|edly)?\b', text, flags=re.I):
        constraints["supervision_hint"] = "supervised"
    elif re.search(r'\bunsupervis(ed|edly)?\b', text, flags=re.I):
        constraints["supervision_hint"] = "unsupervised"

    return constraints


# ---------------------------
# Modality inference & metadata
# ---------------------------
def infer_modality_from_path(path: str):
    p = path.lower()
    if p.endswith(('.csv', '.tsv', '.xlsx', '.xls')):
        # Load small sample to inspect
        try:
            df = pd.read_csv(path, nrows=5) if p.endswith(('.csv', '.tsv')) else pd.read_excel(path, nrows=5)
            cols_lower = [c.lower() for c in df.columns]

            # Heuristic: time series if column names contain time/timestamp/date
            if any("time" in c or "timestamp" in c or "date" in c for c in cols_lower):
                return 'timeseries'
            
            # Optionally: detect if values are mostly numeric → tabular
            numeric_cols = df.select_dtypes(include=np.number).shape[1]
            if numeric_cols / max(1, len(df.columns)) > 0.5:
                return 'tabular'
            
            return 'tabular'
        except Exception as e:
            return 'unknown'
    
    if p.endswith(('.mat', '.npy', '.npz')):
        return 'maybe-mat'
    if p.endswith(('.gpickle', '.gml', '.edgelist')):
        return 'graph'
    if p.endswith(('.wav', '.flac', '.mp3')):
        return 'timeseries'
    if p.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
        return 'image'
    if p.endswith('.json'):
        return 'text'
    return 'unknown'



def load_metadata(path: str):
    meta = {"path": path, "exists": os.path.exists(path), "modality_hint": infer_modality_from_path(path)}
    if not meta["exists"]:
        return meta
    try:
        if path.endswith(('.csv', '.tsv')):
            df = pd.read_csv(path, nrows=20)
            meta["shape"] = df.shape
            meta["sample_columns"] = list(df.columns[:10])
        elif path.endswith('.mat'):
            mat = scipy.io.loadmat(path)
            meta["mat_vars"] = {k: getattr(v, 'shape', None) for k, v in mat.items() if not k.startswith('__')}
        elif path.endswith(('.gpickle', '.gml', '.edgelist')):
            G = nx.read_edgelist(path) if path.endswith('.edgelist') else (
                nx.read_gml(path) if path.endswith('.gml') else nx.read_gpickle(path)
            )
            meta["graph_n_nodes"] = G.number_of_nodes()
            meta["graph_n_edges"] = G.number_of_edges()
            meta["modality_hint"] = "graph"
        elif path.endswith(('.npy', '.npz')):
            arr = np.load(path, allow_pickle=True)
            meta["shape"] = getattr(arr, "shape", None)
    except Exception as e:
        meta["load_error"] = str(e)
    return meta


def infer_supervision(metadata: dict, hints: dict):
    if "supervision_hint" in hints:
        return hints["supervision_hint"]
    if "sample_columns" in metadata:
        for c in metadata["sample_columns"]:
            if c.lower() in ["label", "target", "y", "class"]:
                return "supervised"
    if "mat_vars" in metadata:
        for k in metadata["mat_vars"].keys():
            if any(lbl in k.lower() for lbl in ["label", "y", "target"]):
                return "supervised"
    return "unsupervised"


# ---------------------------
# Gemini disambiguation
# ---------------------------
def gemini_disambiguate(model: str, question: str, candidates: list):
    prompt = f"""
You are a helpful assistant extracting dataset info.

Command: {question}
Candidate paths: {candidates}

Return JSON with:
  - chosen_path
  - reason
"""
    try:
        response = genai.GenerativeModel(model).generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}


# ---------------------------
# Build context
# ---------------------------
def build_context(user_command: str, model: str):
    parsed = parse_command(user_command)
    constraints = extract_constraints(user_command)
    paths = parsed.get("paths") or []

    dataset_metadata = {p: load_metadata(p) for p in paths}

    llm_ans = None
    if not paths:
        llm_ans = gemini_disambiguate(model, user_command, [])

    modalities = [m.get("modality_hint", "unknown") for m in dataset_metadata.values()]
    modality = modalities[0] if modalities else "unknown"


    sample_meta = next(iter(dataset_metadata.values()), {})
    supervision = infer_supervision(sample_meta, constraints)

    return {
        "user_command": user_command,
        "parsed_command": parsed,
        "constraints": constraints,
        "datasets": dataset_metadata,
        "inferred_modality": modality,
        "inferred_supervision": supervision,
        "llm_disambiguation": llm_ans,
        "notes": "Processor agent output (AD-AGENT design)"
    }


# ---------------------------
# Main entry
# ---------------------------
if __name__ == "__main__":
    API_KEY = os.getenv("GEMINI_API_KEY")
    model = setup_gemini(API_KEY)

    # 👉 User enters command
    user_cmd = input("Enter your command: ")

    context = build_context(user_cmd, model)
    print(json.dumps(context, indent=2))
