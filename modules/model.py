# -*- coding: utf-8 -*-
#Author: Liuxin YANG
#Date: 2025-06-06
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from config.configuration import ModelConfig
from modules.nlp import NLPHierarchyClassifier

def train_model(
        cfg: ModelConfig,
        dataset: Dataset, 
        model: nn.Module,
        model_save_name: str,
        load_from_checkpoint: bool = False,
        model_read_path: str = None,
    ) -> nn.Module:

    if load_from_checkpoint:
        model.load_state_dict(torch.load(model_read_path))
    model = model.to(cfg.device)

    dataLoader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for xb,yb in dataLoader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)

            preds = model(xb)
            loss = sum([loss_fn(preds[i], yb[:, i]) for i in range(6)])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{cfg.epochs}, Loss: {total_loss:.4f}")
        # torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch{epoch+1}.pt"))

    torch.save(model.state_dict(), os.path.join(cfg.model_save_path, model_save_name))
    return model

def inference_model(
        cfg: ModelConfig,
        dataset: Dataset,
        input_dim: int,
        n_classes_per_level: list,
        model_read_path: str,
    ) -> np.ndarray:
    
    model = NLPHierarchyClassifier(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        n_classes_per_level=n_classes_per_level
    )

    model.load_state_dict(torch.load(model_read_path, map_location=cfg.device))
    model = model.to(cfg.device)
    model.eval()

    dataLoader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    predictions = []
    with torch.no_grad():
        for xb in dataLoader:
            xb = xb.to(cfg.device)
            preds = model(xb)
            batch_preds = [torch.argmax(pred, dim=1).cpu().numpy() for pred in preds]
            batch_preds = np.stack(batch_preds, axis=1)  # Shape: (batch_size, 6)
            predictions.append(batch_preds)

    pred_labels = np.vstack(predictions)  # Shape: (num_batches, batch_size, 6)
    return pred_labels

def decode_predictions(
    pred_labels: np.ndarray,
    reverse_label_map: dict[str, dict[int, str]],
    label_cols: list[str],
    df: pd.DataFrame = None
) -> pd.DataFrame:
    
    decoded_preds = {
        f"pred_{col}": [reverse_label_map[col][idx] for idx in pred_labels[:, i]]
        for i, col in enumerate(label_cols)
    }

    pred_df = pd.DataFrame(decoded_preds)
    if df is not None:
        return pd.concat([df.reset_index(drop=True), pred_df], axis=1)
    return pred_df