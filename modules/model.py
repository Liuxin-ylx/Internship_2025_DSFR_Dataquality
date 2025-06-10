# -*- coding: utf-8 -*-
#Author: Liuxin YANG
#Date: 2025-06-06
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Type
from torch.utils.data import Dataset, DataLoader

def train_model(
        save_path: str,
        dataset: Dataset, 
        model: nn.Module,
        batch_size: int = 2,
        epochs: int = 5,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> nn.Module:

    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)


    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb,yb in dataLoader:
            xb = xb.to(device)
            yb = yb.to(device)

            preds = model(xb)
            loss = sum([loss_fn(preds[i], yb[:, i]) for i in range(6)])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch{epoch+1}.pt"))

    torch.save(model.state_dict(), os.path.join(save_path, "final_model.pt"))
    return model

def inference_model(
        model_class: Type[nn.Module],
        model_path: str,
        dataset: Dataset,
        input_dim: int,
        batch_size: int,
        n_classes_per_level: list[int],
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    ) -> np.ndarray:
    
    model = model_class(
        input_dim=input_dim,
        hidden_dim=128,  # Assuming a fixed hidden dimension
        n_classes_per_level=n_classes_per_level
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    with torch.no_grad():
        for xb in dataLoader:
            xb = xb.to(device)
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