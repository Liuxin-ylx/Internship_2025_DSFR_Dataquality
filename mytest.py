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

def run(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model_path="../sentence-transformers",
        mode:str = 'train',
        batch_size:int = 2,
        epochs:int = 5,
        hidden_dim:int = 128):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_cols=[
                "hierarchy_level1_desc",
                "global_hierarchy_level2_desc",
                "global_hierarchy_level3_desc",
                "global_hierarchy_level4_desc",
                "global_hierarchy_level5_desc",
                "global_hierarchy_level6_desc"
            ]
    

    # 1.Check dataset
    if train_df is None and test_df is None:
        raise ValueError("Both train_df and test_df cannot be None. Please provide at least one DataFrame.")
    if train_df is None:
        train_df = test_df
    if test_df is None:
        test_df = train_df

    # 2.Load dataset and generate embeddings
    # df = pd.read_csv("data/dataset.csv")
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    df = pd.concat([train_df, test_df], ignore_index=True)
    df["description"] = "Description du produit: " + df["item_desc"] + "local_brand_name: " + df["local_brand_name"] + ", global_brand_name: " + df["global_brand_name"]
        
    X_all, y_all, label_map, reverse_label_map = generate_embedding(
        df=df,
        model_path = model_path,
        text_cols = "description",
        value_cols = None,
        category_cols =["color_desc", "size_desc"],
        label_cols = label_cols
    )
    import pickle
    with open('checkpoints/label_map.pkl', 'wb') as f:
        pickle.dump((label_map, reverse_label_map), f)

    split_mask = df['split'] == 'train'
    X_train, y_train = X_all[split_mask], y_all[split_mask]
    X_test, y_test = X_all[~split_mask], y_all[~split_mask]
    train_df = df[split_mask].copy()
    test_df = df[~split_mask].copy()    

    # dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    n_classes_per_level = [len(df[col].unique()) for col in label_cols]
    if mode == 'train':
        print("Training mode: Training the model...")
        train_dataset = NLPDataset(X_train, y_train, return_labels=True)
        model = NLPHierarchyClassifier(
            input_dim = X_train.shape[1],
            hidden_dim = hidden_dim,
            n_classes_per_level = n_classes_per_level,
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
    elif mode == 'inference':
        print("Inference mode: Loading the model for inference...")
        import pickle
        with open('checkpoints/label_map.pkl', 'rb') as f:
            label_map, reverse_label_map = pickle.load(f)

        test_dataset = NLPDataset(X_test, y_test, return_labels=False)
        pred_labels = inference_model(
            model_class = NLPHierarchyClassifier,
            model_path = "checkpoints/final_model.pt",
            input_dim = X_test.shape[1],
            n_classes_per_level = n_classes_per_level,
            dataset = test_dataset,
            batch_size = 2
        )

        # Print the predicted labels
        decoded_df = decode_predictions(pred_labels, reverse_label_map, label_cols, test_df)
        pd.set_option('display.max_columns', None)
        preds = decoded_df[decoded_df.columns[-6:]]
        preds.to_csv("data/decoded_predictions_v2.csv", index=True)
        print(preds)
    else: 
        raise ValueError("Invalid mode. Choose 'train' or 'inference'.")



result = run(
    train_df=pd.read_csv("data/dataset - train.csv"),
    test_df=pd.read_csv("data/dataset - test.csv"),
    mode='inference',
    batch_size=2,
    epochs=100,
    hidden_dim=128
)