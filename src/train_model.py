"""
Phase 2: Random Forest Wildfire Risk Classifier
Input:  data/firms_enriched.csv  (fire detections + weather)
Output: models/wildfire_rf_model.pkl
        outputs/model_metrics.json
        outputs/feature_importance.png
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, classification_report,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import json
import os

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# REMOVED 'frp' and 'bright_ti4' to prevent data leakage (those are measured *during* a fire)
# We only use weather, location, and seasonality features for risk prediction.
FEATURES = [
    "latitude", "longitude",
    "temperature", "humidity", "wind_speed", "precipitation",
    "month", "day_of_year",
]

def load_and_build_dataset(fire_csv: str, n_fire=5000, n_background=5000):
    """
    Build a binary classification dataset:
      label=1 -> confirmed fire detection from FIRMS
      label=0 -> random background point (sampled from CA bounding box, no fire)
    """
    print("Loading enriched fire detections...")
    fire_df = pd.read_csv(fire_csv)
    fire_df["acq_date"]    = pd.to_datetime(fire_df["acq_date"])
    fire_df["month"]       = fire_df["acq_date"].dt.month
    fire_df["day_of_year"] = fire_df["acq_date"].dt.dayofyear
    fire_df["label"]       = 1

    # Sample n_fire positive examples
    n_fire = min(n_fire, len(fire_df))
    fire_sample = fire_df.sample(n=n_fire, random_state=42)

    # Generate n_background negative (non-fire) background points
    print(f"Generating {n_background} background (non-fire) points...")
    np.random.seed(99)
    bg_lats = np.random.uniform(32.5, 42.0, n_background)
    bg_lons = np.random.uniform(-124.5, -114.0, n_background)
    bg_months = np.random.randint(1, 13, n_background)
    bg_doy    = np.random.randint(1, 366, n_background)

    # Assign weather from closest fire detection (median by month as proxy)
    monthly_stats = fire_df.groupby("month")[["temperature","humidity","wind_speed","precipitation"]].median()
    
    bg_records = []
    for i in range(n_background):
        m = bg_months[i]
        stats = monthly_stats.loc[m] if m in monthly_stats.index else monthly_stats.median()
        bg_records.append({
            "latitude":     bg_lats[i],
            "longitude":    bg_lons[i],
            "temperature":  stats["temperature"]  + np.random.normal(0, 3),
            "humidity":     min(100, stats["humidity"]    + np.random.normal(0, 10)),
            "wind_speed":   max(0,   stats["wind_speed"]   + np.random.normal(0, 5)),
            "precipitation":max(0,   stats["precipitation"] + np.random.normal(0, 2)),
            "month":        m,
            "day_of_year":  bg_doy[i],
            "label":        0
        })
    bg_df = pd.DataFrame(bg_records)

    dataset = pd.concat([
        fire_sample[FEATURES + ["label"]],
        bg_df[FEATURES + ["label"]]
    ], ignore_index=True).dropna()

    print(f"Dataset: {len(dataset)} samples  ({dataset['label'].sum()} fire, {(dataset['label']==0).sum()} background)")
    return dataset

def spatial_train_test_split(df):
    """
    Geographic split to avoid data leakage:
    - Train: Northern CA  (lat >= 36.5)
    - Test:  Southern CA  (lat <  36.5)
    This tests genuine generalisation to unseen regions.
    """
    train = df[df["latitude"] >= 36.5].copy()
    test  = df[df["latitude"] <  36.5].copy()
    print(f"Spatial split - Train (N CA): {len(train)}, Test (S CA): {len(test)}")
    return train, test

def train_and_evaluate(train, test):
    X_train = train[FEATURES]
    y_train = train["label"]
    X_test  = test[FEATURES]
    y_test  = test["label"]

    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc       = roc_auc_score(y_test, y_proba)

    metrics = {
        "precision":      round(precision, 4),
        "recall":         round(recall, 4),
        "f1_score":       round(f1, 4),
        "roc_auc":        round(auc, 4),
        "train_samples":  int(len(train)),
        "test_samples":   int(len(test)),
        "n_features":     len(FEATURES),
        "n_estimators":   200,
        "spatial_split":  "N CA train (lat>=36.5) / S CA test (lat<36.5)",
        "data_source":    "NASA FIRMS VIIRS 375m (2019-2023)",
    }
    print("\n--- Model Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n" + classification_report(y_test, y_pred, target_names=["No Fire","Fire"]))
    return rf, metrics

def plot_importance(rf, feature_names, out_path):
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#1e293b")
    fig.patch.set_facecolor("#0f172a")
    ax.barh(
        [feature_names[i] for i in idx[::-1]],
        importances[idx[::-1]],
        color="#3b82f6"
    )
    ax.set_xlabel("Importance", color="white")
    ax.set_title("Random Forest - Feature Importance\n(Wildfire Risk Classifier)", color="white")
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Feature importance saved -> {out_path}")

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2: Random Forest Training")
    print("=" * 60)
    
    enriched_path = os.path.join(DATA_DIR, "firms_enriched.csv")
    if not os.path.exists(enriched_path):
        print(f"ERROR: {enriched_path} not found.")
        exit(1)
    
    dataset = load_and_build_dataset(enriched_path, n_fire=5000, n_background=5000)
    train, test = spatial_train_test_split(dataset)
    rf, metrics = train_and_evaluate(train, test)

    # Save model
    model_out = os.path.join(MODELS_DIR, "wildfire_rf_model.pkl")
    with open(model_out, "wb") as f:
        pickle.dump(rf, f)
    print(f"\nModel saved -> {model_out}")

    # Save metrics
    metrics_out = os.path.join(OUTPUTS_DIR, "model_metrics.json")
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> {metrics_out}")

    # Plot feature importance
    img_out = os.path.join(OUTPUTS_DIR, "feature_importance.png")
    plot_importance(rf, FEATURES, img_out)
    
    print("\n✓ Phase 2 complete. Run src/analysis.py to score bonds with the ML model.")
