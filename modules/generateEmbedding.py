# -*- coding: utf-8 -*-
#Author: Liuxin YANG
#Date: 2025-05-31

import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from config.configuration import ModelConfig


def generate_embedding(cfg: ModelConfig, df: pd.DataFrame, model_path, text_cols: str) -> None:
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
    if cfg.value_cols is not None:
        scaler = StandardScaler()
        for col in cfg.value_cols:
            value_scaled.append(scaler.fit_transform(df[[col]]))
        value_scaled = np.hstack(value_scaled)
    else:
        value_scaled = np.array([])

    category_onehot = []
    if cfg.category_cols is not None:
        for col in cfg.category_cols:
            onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            category_onehot.append(onehot.fit_transform(df[[col]].astype(str)))
        category_onehot = np.hstack(category_onehot)
    else:
        category_onehot = np.array([])

    feature_parts = []
    for part in [desc_embeddings, value_scaled, category_onehot]:
        if isinstance(part, (np.ndarray, torch.Tensor)) and part.size > 0:
            feature_parts.append(part)
    X = np.hstack(feature_parts)

    label_map = {}
    reverse_label_map = {}
    y = None
    if cfg.hierarchy_cols is not None:
        for col in cfg.hierarchy_cols:
            if col not in df.columns:
                raise ValueError(f"Label column '{col}' not found in DataFrame.")
            
            label_map[col] = {v: i for i, v in enumerate(df[col].unique())}
            reverse_label_map[col] = {v: k for k, v in label_map[col].items()}

            df[col + "_idx"] = df[col].map(label_map[col])
        y = df[[col + "_idx" for col in cfg.hierarchy_cols]].values
    else:
        y = None

    ylabels = None
    if cfg.correct_hierarchy_cols is not None:
        for col in cfg.correct_hierarchy_cols:
            if col not in df.columns:
                raise ValueError(f"Correct label column '{col}' not found in DataFrame.")
        
            df[col + "_idx"] = df[col].map(label_map[col.replace("correct_", "")])
        ylabels = df[[col + "_idx" for col in cfg.correct_hierarchy_cols]].values
    else:
        ylabels = None

    label = None
    if cfg.label_cols is not None:
        for col in cfg.label_cols:
            if col not in df.columns:
                raise ValueError(f"Label column '{col}' not found in DataFrame.")
        
            label_map[col] = {v: i for i, v in enumerate(df[col].unique())}
            reverse_label_map[col] = {v: k for k, v in label_map[col].items()}
        label = df[cfg.label_cols].astype(bool).values
    else:
        label = None

    return X, y, ylabels, label, label_map, reverse_label_map