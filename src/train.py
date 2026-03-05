"""
Main training pipeline script.
Loads config, fetches data, engineers features, splits validation, and trains LightGBM.
"""
import os
import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

from src.data_loader import get_processed_dataframe
from src.features import build_features
from src.splits import PanelTimeSeriesSplit

def seed_everything(seed=42):
    """Locks seeds for reproducibility."""
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "configs", "model_config.yaml")
    config = load_config(config_path)
    
    # 1. Initialization
    # We ignore the XGBoost params from the old config and use robust LightGBM defaults
    seed = 42
    seed_everything(seed)
    
    # 2. Data & Features
    print("Loading raw data...")
    raw_dir = os.path.join(base_dir, "data", "raw")
    df = get_processed_dataframe(raw_dir, start_date=config["data"]["start_date"])
    
    print("Building features...")
    feat_df = build_features(df)
    
    # Sort index properly for TimeSeriesSplit
    feat_df = feat_df.sort_index()

    feature_cols = config["features"]["feature_cols"]
    target_col = config["features"]["target_col"]
    
    # Check if features exist in dataframe
    # We added new volume features in features.py, let's include them if available
    available_features = [c for c in feature_cols if c in feat_df.columns]
    
    # Also add the new volume features built earlier if they exist
    if "volume_change_1d" in feat_df.columns and "volume_change_1d" not in available_features:
        feat_df["volume_change_1d"] = feat_df["volume_change_1d"].astype(float)
        available_features.append("volume_change_1d")
    if "volume_ma_10_ratio" in feat_df.columns and "volume_ma_10_ratio" not in available_features:
        feat_df["volume_ma_10_ratio"] = feat_df["volume_ma_10_ratio"].astype(float)
        available_features.append("volume_ma_10_ratio")

    missing_cols = [c for c in feature_cols if c not in available_features]
    if missing_cols:
        print(f"Warning: Missing configured features: {missing_cols}")

    # Drop NaNs
    valid_df = feat_df.dropna(subset=available_features + [target_col, "target_return"])

    # X, y data
    X = valid_df[available_features]
    y = valid_df[target_col]

    print(f"Dataset shape after NaN drop: {X.shape}")

    # 3. Validation Split & Train
    print("Initializing Time Series Panel Split...")
    n_splits = config["validation"]["n_splits"]
    gap = config["validation"]["gap"]
    splitter = PanelTimeSeriesSplit(n_splits=n_splits, gap=gap)
    
    metrics = []
    
    # Track model for feature importance
    last_model = None
    last_X_train_cols = None
    
    for fold, (train_idx, test_idx) in enumerate(splitter.split(valid_df)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        tr_dates = valid_df.iloc[train_idx].index.get_level_values("date")
        te_dates = valid_df.iloc[test_idx].index.get_level_values("date")
        
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        print(f"Train: {tr_dates.min().date()} to {tr_dates.max().date()} ({len(X_train)} samples)")
        print(f"Test : {te_dates.min().date()} to {te_dates.max().date()} ({len(X_test)} samples)")
        
        # Calculate sample weights: absolute return magnitude
        # This prevents the model from treating a 0.01% day the same as a 5% day
        train_weights = abs(valid_df.iloc[train_idx]["target_return"])
        
        # Train LightGBM model
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1
        )
        
        model.fit(
            X_train, 
            y_train,
            sample_weight=train_weights
        )
        
        last_model = model
        last_X_train_cols = X_train.columns
        
        # Predict
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        try:
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            auc = roc_auc_score(y_test, probs)
        except ValueError as e:
            print(f"Error computing metrics: {e}")
            acc, prec, auc = 0, 0, 0
        
        print(f"Fold {fold+1} Metrics: Accuracy={acc:.4f}, Precision={prec:.4f}, AUC={auc:.4f}")
        
        metrics.append({
            "fold": fold + 1,
            "accuracy": acc,
            "precision": prec,
            "auc": auc
        })
        
        # Keep last fold for backtest
        if fold == n_splits - 1:
            out_df = pd.DataFrame(index=X_test.index)
            out_df["target"] = y_test
            out_df["pred_prob"] = probs
            out_df["pred"] = preds
            out_df["target_return"] = valid_df.iloc[test_idx]["target_return"]
            
            res_dir = os.path.join(base_dir, "results", "metrics")
            os.makedirs(res_dir, exist_ok=True)
            out_path = os.path.join(res_dir, "oof_predictions.csv")
            out_df.to_csv(out_path)
            print(f"Saved out-of-sample predictions to {out_path}")

    # Summary
    avg_metrics = pd.DataFrame(metrics).mean()
    print("\n--- Average Walk-Forward Metrics ---")
    print(avg_metrics[["accuracy", "precision", "auc"]].to_string())
    
    # Feature Importance Plot (using the final fold model)
    print("\nGenerating Feature Importance Plot...")
    fig_dir = os.path.join(base_dir, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    importances = last_model.feature_importances_
    
    # Sort indices by importance
    sorted_idx = np.argsort(importances)
    sorted_features = [last_X_train_cols[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]
    
    plt.figure(figsize=(10, 8))
    plt.barh(sorted_features, sorted_importances, color='skyblue')
    plt.title("LightGBM Feature Importance (Final Fold)")
    plt.xlabel("Importance (Split Count)")
    plt.tight_layout()
    
    fig_path = os.path.join(fig_dir, "feature_importance.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved feature importance plot to {fig_path}")

if __name__ == "__main__":
    run_pipeline()