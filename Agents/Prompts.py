def generate_model_selection_prompt_from_timeseries(name, size, dim, type):

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
        - Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting (Autoformer)
        - Are Transformers Effective for Time Series Forecasting? (DLinear)
        - Exponential Smoothing Transformers for Time-series Forecasting (ETSformer)
        - Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting (FEDformer)
        - Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (Informer)
        - Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures (LightTS)
        - Low-complexity Pyramidal Attention for Long-range Time Series Modeling and Forecasting (Pyraformer)
        - The Efficient Transformer (Reformer)
        - Temporal 2D-Variation Modeling for General Time Series Analysis (TimesNet)
        - Attention is All You Need (Transformer)

        ## Rules:
        1. Availabel options include "Autoformer", "DLinear", "ETSformer", "FEDformer", "Informer", "LightTS", "Pyraformer", "Reformer", "TimesNet", and "Transformer."
        2. Treat all models equally and evaluate them based on their compatibility with the dataset characteristics and the anomaly detection task.
        3. Response Format:
            - Provide responses in a strict **JSON** format with the keys "reason" and "choice."
                - "reason": Your explanation of the reasoning.
                - "choice": The model you have selected for anomaly detection in this dataset.

        Response in JSON format:
        """

    messages = [
        # {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        # {"role": "assistant", "content": assistant_message}
    ]

    return messages




def generate_model_selection_prompt_from_pyod(name, size, dim):

    user_message = f"""
        You are an expert in model selection for anomaly detection on multivariate data.

        ## Task:
        - Given the information of a dataset and a set of models, select the model you believe will achieve the best performance for detecting anomalies in this dataset. Provide a brief explanation of your choice.

        ## Dataset Information:
        - Dataset Name: {name}
        - Dataset Size: {size}
        - Data Dimension: {dim}

        ## Model Options:
        - Adversarially Learned Anomaly Detection (ALAD)
        - Anomaly Detection with Generative Adversarial Networks (AnoGAN)
        - AutoEncoder (AE)
        - Autoencoder-based One-class Support Vector Machine (AE1SVM)
        - Deep One-Class Classification (DeepSVDD)
        - Deep Anomaly Detection with Deviation Networks (DevNet)
        - Unifying Local Outlier Detection Methods via Graph Neural Networks (LUNAR)
        - Multiple-Objective Generative Adversarial Active Learning (MO-GAAL)
        - Single-Objective Generative Adversarial Active Learning (SO-GAAL)
        - Variational AutoEncoder (VAE)

        ## Rules:
        1. Availabel options include "ALAD", "AnoGAN", "AE", "AE1SVM", "DeepSVDD", "DevNet", "LUNAR", "MO-GAAL", "SO-GAAL", and "VAE."
        2. Treat all models equally and evaluate them based on their compatibility with the dataset characteristics and the anomaly detection task.
        3. Response Format:
            - Provide responses in a strict **JSON** format with the keys "reason" and "choice."
                - "reason": Your explanation of the reasoning.
                - "choice": The model you have selected for anomaly detection in this dataset.

        Response in JSON format:
        """
    
    messages = [
        # {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        # {"role": "assistant", "content": assistant_message}
    ]

    return messages



def generate_model_selection_prompt_from_pygod(name, num_node, num_edge, num_feature, avg_degree):

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
        - Adversarial Outlier Aware Attributed Network Embedding (AdONE)
        - A Joint Modeling Approach for Anomaly Detection on Attributed Networks (ANOMALOUS)
        - Dual Autoencoder for Anomaly Detection on Attributed Networks (AnomalyDAE)
        - Contrastive Attributed Network Anomaly Detection (CONAD)
        - Deep Outlier Aware Attributed Network Embedding (DONE)
        - Generative Adversarial Attributed Network Anomaly Detection (GAAN)
        - Higher-order Structure based Anomaly Detection on Attributed Networks (GUIDE)
        - Residual Analysis for Anomaly Detection in Attributed Networks (Radar)
        - Structural Clustering Algorithm for Networks (SCAN)
        
        ## Rules:
        1. Availabel options include "AdONE", "ANOMALOUS", "AnomalyDAE", "CONAD", "DONE", "GAAN", "GUIDE", "Radar", and "SCAN."
        2. Treat all models equally and evaluate them based on their compatibility with the dataset characteristics and the anomaly detection task.
        3. Response Format:
            - Provide responses in a strict **JSON** format with the keys "reason" and "choice."
                - "reason": Your explanation of the reasoning.
                - "choice": The model you have selected for anomaly detection in this dataset.
        
        Response in JSON format:
        """
    
    messages = [
        # {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        # {"role": "assistant", "content": assistant_message}
    ]

    return messages