# -*- coding: utf-8 -*-
#Author: Liuxin YANG
#Date: 2025-05-31

import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def generate_embedding(df: pd.DataFrame, model_path, text_cols, value_cols, category_cols, label_cols: str) -> None:
    st_model = SentenceTransformer(model_path)
    
    if text_cols is not None:
        desc_embeddings = st_model.encode(
            df[text_cols].fillna(""),
            convert_to_tensor=False, 
            batch_size=8,
            device="cuda" if st_model.device.type == "cuda" else "cpu",
            show_progress_bar=True
        )
    else:
        desc_embeddings = np.array([])
    
    value_scaled = []
    if value_cols is not None:
        scaler = StandardScaler()
        for col in value_cols:
            value_scaled.append(scaler.fit_transform(df[[col]]))
        value_scaled = np.hstack(value_scaled)
    else:
        value_scaled = np.array([])

    category_onehot = []
    if category_cols is not None:
        for col in category_cols:
            onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            category_onehot.append(onehot.fit_transform(df[[col]]))
        category_onehot = np.hstack(category_onehot)
    else:
        category_onehot = np.array([])

    y = None
    if label_cols is not None:
        for col in label_cols:
            if col not in df.columns:
                raise ValueError(f"Label column '{col}' not found in DataFrame.")
            
        label_map = {col: {v: i for i, v in enumerate(df[col].unique())} for col in label_cols}
        reverse_label_map = {col: {v: k for k, v in label_map[col].items()} for col in label_map}

        for col in label_cols:
            df[col + "_idx"] = df[col].map(label_map[col])
        y = df[[col + "_idx" for col in label_cols]].values

    feature_parts = []
    for part in [desc_embeddings, value_scaled, category_onehot]:
        if isinstance(part, (np.ndarray, torch.Tensor)) and part.size > 0:
            feature_parts.append(part)
    X = np.hstack(feature_parts)


    return X, y, label_map, reverse_label_map