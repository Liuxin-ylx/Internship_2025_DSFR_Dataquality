import pandas as pd
import torch
from torch.utils.data import DataLoader

from modules.generateEmbedding import generate_embedding
from modules.nlp import NLPDataset, NLPHierarchyClassifier
from modules.model import (
    train_model, 
    inference_model,
    decode_predictions
)

batch_size = 2
epochs = 5
hidden_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load dataset and generate embeddings
df = pd.read_csv("data/dataset.csv")
df["description"] = "Description du produit:" + df["item_desc"]+ "local_brand_name: " + df["local_brand_name"] + ", global_brand_name: " + df["global_brand_name"]
label_cols=[
        "hierarchy_level1_desc",
        "local_hierarchy_level2_desc",
        "local_hierarchy_level3_desc",
        "local_hierarchy_level4_desc",
        "local_hierarchy_level5_desc",
        "local_hierarchy_level6_desc"
    ]

X, y, label_map, reverse_label_map = generate_embedding(
    df=df,
    model_path="../sentence-transformers",
    text_cols="description",
    value_cols= None,
    category_cols=["color", "size"],
    label_cols=label_cols
)
n_classes_per_level = [len(df[col].unique()) for col in label_cols]

# Create dataset and dataloader
train_dataset = NLPDataset(X, y, return_labels=True)
dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = NLPHierarchyClassifier(
    input_dim=X.shape[1],
    hidden_dim=hidden_dim,
    n_classes_per_level=n_classes_per_level 
).to(device)

# Train the model
trained_model = train_model(
    save_path="checkpoints/",
    dataset=train_dataset,
    model=model,
    batch_size=batch_size,
    epochs=epochs,
    device=device
)

# Inference
test_dataset = NLPDataset(X, y, return_labels=False)
pred_labels = inference_model(
    model_class=NLPHierarchyClassifier,
    model_path="checkpoints/final_model.pt",
    input_dim=X.shape[1],
    n_classes_per_level=n_classes_per_level,
    dataset=test_dataset,
    batch_size=2
)

# Print the predicted labels
decoded_df = decode_predictions(pred_labels, reverse_label_map, label_cols, df)
pd.set_option('display.max_columns', None)
print(decoded_df)
#decoded_df.to_csv("decoded_predictions.csv", index=False)