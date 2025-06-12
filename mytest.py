import pickle
import pandas as pd
from torch.utils.data import DataLoader

from modules.generateEmbedding import generate_embedding
from modules.nlp import NLPDataset, NLPHierarchyClassifier
from modules.model import (
    train_model, 
    inference_model,
    decode_predictions
)
from config.configuration import ModelConfig
def run(
        cfg: ModelConfig,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        mode: str = 'train',
        load_from_checkpoint: bool = True,
        model_save_name = "final_model.pt",
        model_read_path = "checkpoints/final_model.pt"):
    
    # 1.Check dataset
    if train_df is None and test_df is None:
        raise ValueError("Both train_df and test_df cannot be None. Please provide at least one DataFrame.")
    if train_df is None:
        train_df = test_df
    if test_df is None:
        test_df = train_df

    # 2.Load dataset and generate embeddings
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    df = pd.concat([train_df, test_df], ignore_index=True)
    df["description"] = "Description du produit: " + df["item_desc"] + "local_brand_name: " + df["local_brand_name"] + ", global_brand_name: " + df["global_brand_name"]
        
    X_all, y_all, label_map, reverse_label_map = generate_embedding(
        df=df,
        model_path = cfg.embedding_path,
        text_cols = "description",
        value_cols = cfg.value_cols,
        category_cols = cfg.category_cols,
        label_cols = cfg.label_cols
    )
    with open('checkpoints/label_map.pkl', 'wb') as f:
        pickle.dump((label_map, reverse_label_map), f)

    split_mask = df['split'] == 'train'
    X_train, y_train = X_all[split_mask], y_all[split_mask]
    X_test, y_test = X_all[~split_mask], y_all[~split_mask]
    train_df = df[split_mask].copy()
    test_df = df[~split_mask].copy()    

    # dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    n_classes_per_level = [len(df[col].unique()) for col in cfg.label_cols]
    if mode == 'train':
        print("Training mode: Training the model...")
        train_dataset = NLPDataset(X_train, y_train, return_labels=True)
        model = NLPHierarchyClassifier(
            input_dim = X_train.shape[1],
            hidden_dim = cfg.hidden_dim,
            n_classes_per_level = n_classes_per_level,
        ).to(cfg.device)

        # Train the model
        trained_model = train_model(
            cfg = cfg,
            dataset = train_dataset,
            model = model,
            model_save_name = model_save_name,
            load_from_checkpoint = load_from_checkpoint,
            model_read_path = model_read_path
        )
    # Inference
    elif mode == 'inference':
        print("Inference mode: Loading the model for inference...")
        with open('checkpoints/label_map.pkl', 'rb') as f:
            label_map, reverse_label_map = pickle.load(f)

        test_dataset = NLPDataset(X_test, y_test, return_labels=False)

        pred_labels = inference_model(
            cfg = cfg,
            dataset = test_dataset,
            input_dim = X_test.shape[1],
            n_classes_per_level = n_classes_per_level,
            model_read_path = model_read_path
        )

        # Print the predicted labels
        decoded_df = decode_predictions(pred_labels, reverse_label_map, cfg.label_cols, test_df)
        pd.set_option('display.max_columns', None)
        preds = decoded_df[decoded_df.columns[-6:]]
        preds.to_csv("data/decoded_predictions_v2.csv", index=True)
        print(preds)
    else: 
        raise ValueError("Invalid mode. Choose 'train' or 'inference'.")


cfg = ModelConfig()
result = run(
    cfg,
    train_df=pd.read_csv("data/dataset - train.csv"),
    test_df=pd.read_csv("data/dataset - test.csv"),
    mode='inference',
    load_from_checkpoint=True,
    model_save_name = "model_latest2.pt",
    model_read_path = "checkpoints/model_latest2.pt"
)