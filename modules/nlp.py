# -*- coding: utf-8 -*-
#Author: Liuxin YANG
#Date: 2025-05-31

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class NLPDataset(Dataset):
    def __init__(self, X, y, return_labels=True):
        self.X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        self.y = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)
        self.return_labels = return_labels

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.return_labels:
            return self.X[idx],self.y[idx]
            
        else:
            return self.X[idx]
            

class NLPHierarchyClassifier(nn.Module):
    def __init__(
            self, 
            input_dim,
            hidden_dim,
            n_classes_per_level
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, n_classes) for n_classes in n_classes_per_level
        ])

    def forward(self, x):
        output = self.encoder(x)
        return [head(output) for head in self.heads]
    
    