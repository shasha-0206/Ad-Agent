def generate_model_selection_prompt_from_timeseries(name, size, dim, type):

    system_message = "You MUST reply ONLY in valid JSON with keys 'reason' and 'choice'. Do not add extra text or explanations outside JSON."

    user_message = f"""
You are an expert in model selection for anomaly detection on time series data.

## Task:
- Given the information of a dataset and a set of models, select the model you believe will achieve the best performance for detecting anomalies in this dataset. Provide a brief explanation of your choice.

## Dataset Information:
- Dataset Name: {name}
- Dataset Size: {size}
- Data Dimension: {dim}
- Data Type: {type}

## Model Options:
- Autoformer
- DLinear
- ETSformer
- FEDformer
- Informer
- LightTS
- Pyraformer
- Reformer
- TimesNet
- Transformer

## Rules:
1. Available options include "Autoformer", "DLinear", "ETSformer", "FEDformer", "Informer", "LightTS", "Pyraformer", "Reformer", "TimesNet", and "Transformer."
2. Treat all models equally and evaluate them based on their compatibility with the dataset characteristics and the anomaly detection task.
3. Response Format:
   - Provide responses in strict JSON with keys "reason" and "choice".
   - Example Response:
     {{
        "choice": "TimesNet",
        "reason": "TimesNet works well for multivariate time series with complex temporal dependencies..."
     }}

Respond ONLY in valid JSON.
"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    return messages



def generate_model_selection_prompt_from_pyod(name, size, dim):

    system_message = "You MUST reply ONLY in valid JSON with keys 'reason' and 'choice'. Do not add extra text or explanations outside JSON."

    user_message = f"""
You are an expert in model selection for anomaly detection on multivariate data.

## Task:
- Given the information of a dataset and a set of models, select the model you believe will achieve the best performance for detecting anomalies in this dataset. Provide a brief explanation of your choice.

## Dataset Information:
- Dataset Name: {name}
- Dataset Size: {size}
- Data Dimension: {dim}

## Model Options:
- ALAD
- AnoGAN
- AE
- AE1SVM
- DeepSVDD
- DevNet
- LUNAR
- MO-GAAL
- SO-GAAL
- VAE

## Rules:
1. Available options include "ALAD", "AnoGAN", "AE", "AE1SVM", "DeepSVDD", "DevNet", "LUNAR", "MO-GAAL", "SO-GAAL", and "VAE."
2. Treat all models equally and evaluate them based on their compatibility with the dataset characteristics and the anomaly detection task.
3. Response Format:
   - Provide responses in strict JSON with keys "reason" and "choice".
   - Example Response:
     {{
        "choice": "VAE",
        "reason": "VAE is suitable for small to medium tabular datasets with continuous features and unsupervised anomaly detection."
     }}

Respond ONLY in valid JSON.
"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    return messages



def generate_model_selection_prompt_from_pygod(name, num_node, num_edge, num_feature, avg_degree):

    system_message = "You MUST reply ONLY in valid JSON with keys 'reason' and 'choice'. Do not add extra text or explanations outside JSON."

    user_message = f"""
You are an expert in model selection for anomaly detection on graph data.

## Task:
- Given the information of a dataset and a set of models, select the model you believe will achieve the best performance for detecting anomalies in this dataset. Provide a brief explanation of your choice.

## Dataset Information:
- Dataset Name: {name}
- Number of Nodes: {num_node}
- Number of Edges: {num_edge}
- Number of Features: {num_feature}
- Average Degree: {avg_degree}

## Model Options:
- AdONE
- ANOMALOUS
- AnomalyDAE
- CONAD
- DONE
- GAAN
- GUIDE
- Radar
- SCAN

## Rules:
1. Available options include "AdONE", "ANOMALOUS", "AnomalyDAE", "CONAD", "DONE", "GAAN", "GUIDE", "Radar", and "SCAN."
2. Treat all models equally and evaluate them based on their compatibility with the dataset characteristics and the anomaly detection task.
3. Response Format:
   - Provide responses in strict JSON with keys "reason" and "choice".
   - Example Response:
     {{
        "choice": "CONAD",
        "reason": "CONAD is effective for attributed networks with moderate node and edge density for unsupervised anomaly detection."
     }}

Respond ONLY in valid JSON.
"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    return messages
