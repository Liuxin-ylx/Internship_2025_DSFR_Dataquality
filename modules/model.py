# -*- coding: utf-8 -*-
#Author: Liuxin YANG
#Date: 2025-06-16
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

    # X, y, ylabels, label
    dataLoader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    coef_error = 5
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for xb,yb,ylb,lb in dataLoader:
            xb = xb.to(cfg.device) 
            yb = yb.to(cfg.device) # dim: [Batch_size, len(hierarchy_cols)]
            ylb = ylb.to(cfg.device) # dim: [Batch_size, len(correct_hierarchy_cols)]
            lb = lb.to(cfg.device) # dim: [Batch_size, 1] (is_error label)

            preds = model(xb) # len(hierarchy_cols) outputs, [level1, ..., level6]
            is_error = lb.view(-1).bool()  # dim: [Batch_size]
            
            level_loss = 0.0
            
            for i in range(len(preds)-1):
                if (~is_error).any():
                    level_loss += loss_fn(preds[i][~is_error], yb[:,i][~is_error])
                else:
                    level_loss += loss_fn(preds[i][is_error], ylb[:,i][is_error])
                
            # preds[-1]: logits, dim [batch_size, 2] 
            # lb.long().squeeze(-1): [1,1] <--> [True, True]
            
            label_loss = loss_fn(preds[-1], lb.long().squeeze(-1)) * coef_error  # is_error loss with higher weight

            loss = level_loss + label_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{cfg.epochs}, Loss: {total_loss:.4f}")
        # torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch{epoch+1}.pt"))

    os.makedirs(cfg.model_save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg.model_save_path, model_save_name))
    return model

def inference_model(
        cfg: ModelConfig,
        dataset: Dataset,
        input_dim: int,
        n_classes_per_level: list,
        model_read_path: str,
    ) -> tuple[float, np.ndarray]:
    
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

    accuracy = 0.0
    accuracy_scores = []
    with torch.no_grad():
        if dataset.return_labels == True:
            print("Dataset is set to return category as False. We will not calculate the accuracy. Instead, we will return the predictions directly.")
        
            for xb,yb,ylb,lb in dataLoader:
                xb = xb.to(cfg.device) 
                yb = yb.to(cfg.device) # dim: [Batch_size, len(hierarchy_cols)]
                ylb = ylb.to(cfg.device) # dim: [Batch_size, len(correct_hierarchy_cols)]
                lb = lb.to(cfg.device) # dim: [Batch_size, 1] (is_error label)

                preds = model(xb)
                batch_preds = [torch.argmax(pred, dim=1).cpu().numpy() for pred in preds]
                batch_preds = np.stack(batch_preds, axis=1)  # Shape: (batch_size, 7)
                predictions.append(batch_preds)

                pred_is_error = torch.argmax(preds[-1], dim=1)
                correct = (pred_is_error == lb.squeeze(-1)).float()
                accuracy_scores.append(correct.cpu())
            
            accuracy = torch.cat(accuracy_scores).mean().item()  # Calculate accuracy from the is_error predictions
            pred_labels = np.vstack(predictions)
        else:
            print("Dataset is set to return category as True. We will calculate the accuracy and return the predictions.")
            for xb,yb in dataLoader:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)

                preds = model(xb)
                batch_preds = [torch.argmax(pred, dim=1).cpu().numpy() for pred in preds]
                batch_preds = np.stack(batch_preds, axis=1)  # Shape: (batch_size, 6)
                predictions.append(batch_preds)

            pred_labels = np.vstack(predictions)  # Shape: (num_batches, batch_size, 6)


    return accuracy, pred_labels






def decode_predictions(
    pred_labels: np.ndarray,
    reverse_label_map: dict[str, dict[int, str]],
    hierarchy_cols: list[str],
    df: pd.DataFrame = None
) -> pd.DataFrame:
    
    decoded_preds = {
        f"pred_{col}": [reverse_label_map[col][idx] for idx in pred_labels[:, i]]
        for i, col in enumerate(hierarchy_cols)
    }

    pred_df = pd.DataFrame(decoded_preds)
    if df is not None:
        return pd.concat([df.reset_index(drop=True), pred_df], axis=1)
    return pred_df