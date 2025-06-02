import pandas as pd
import torch
from torch.utils.data import DataLoader
from modules.generateEmbedding import generate_embedding
from modules.nlp import NLPDataset, NLPHierarchyClassifier

df = pd.read_csv("data/dataset.csv")
df["description"] = "Produit description:" + df["item_desc"]+ "local_brand_name: " + df["local_brand_name"] + ", global_brand_name: " + df["global_brand_name"]
X, y = generate_embedding(
    df=df,
    model_path="../sentence-transformers",
    text_cols="description",
    value_cols= None,
    category_cols=["color", "size"],
    label_cols=[
        "hierarchy_level1_desc",
        "local_hierarchy_level2_desc",
        "local_hierarchy_level3_desc",
        "local_hierarchy_level4_desc",
        "local_hierarchy_level5_desc",
        "local_hierarchy_level6_desc"
    ]
)

batch_size = 2
epoche = 5
hidden_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = NLPDataset(X, y)
dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

n_classes_per_level = [len(df[col].unique()) for col in [
    "hierarchy_level1_desc",
    "local_hierarchy_level2_desc",
    "local_hierarchy_level3_desc",
    "local_hierarchy_level4_desc",
    "local_hierarchy_level5_desc",
    "local_hierarchy_level6_desc"
]]

model = NLPHierarchyClassifier(
    input_dim=X.shape[1],
    hidden_dim=hidden_dim,
    n_classes_per_level=n_classes_per_level 
).to(device)


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epoche):
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
    print(f"Epoch {epoch + 1}/{epoche}, Loss: {total_loss:.4f}")