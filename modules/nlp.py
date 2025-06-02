# -*- coding: utf-8 -*-
#Author: Liuxin YANG
#Date: 2025-05-31

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class NLPDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(
            self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long
        )

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
    
    