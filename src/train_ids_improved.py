from pathlib import Path
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier


# =========================
# CONFIG
# =========================
#BASE_DIR = Path(r"C:\Users\rahmy\Desktop\ELTE\UNI 6\Diploma\IDS2\IDS_NORMAL")
BASE_DIR = Path(__file__).resolve().parent
CICIDS_FILE = BASE_DIR / "data" / "processed" / "cicids2017_binary.csv"
CUSTOM_FILE = BASE_DIR / "custom_data" / "processed" / "custom_binary_dataset.csv"
FEATURES_FILE = BASE_DIR / "results" / "training_feature_columns.csv"

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Keep training manageable on Windows
USE_CICIDS_SAMPLE = True
CICIDS_SAMPLE_SIZE = 200_000

# Balance the custom part
USE_CUSTOM_BALANCING = True

# If True, custom attack rows will be sampled to match:
#   custom_attack_sample_size = custom_normal_rows * CUSTOM_ATTACK_NORMAL_RATIO
CUSTOM_ATTACK_NORMAL_RATIO = 1.0

# Optional hard cap on sampled custom attack rows
MAX_CUSTOM_ATTACK_ROWS = 10_000


# =========================
# HELPERS
# =========================
def plot_conf_matrix(cm, title, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Normal (0)", "Attack (1)"])
    plt.yticks(tick_marks, ["Normal (0)", "Attack (1)"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def evaluate_model(model_name, y_true, y_pred, train_seconds):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    metrics = {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "Train_Seconds": train_seconds,
    }

    return metrics, cm, report


def load_feature_columns():
    df = pd.read_csv(FEATURES_FILE)
    return df["feature_name"].tolist()


# =========================
# LOAD CICIDS2017
# =========================
print("Loading CICIDS2017 dataset...")
cicids_df = pd.read_csv(CICIDS_FILE, low_memory=False)

if "Label_Binary" not in cicids_df.columns:
    raise ValueError("Label_Binary column not found in CICIDS file.")

feature_columns = load_feature_columns()

# Keep only the training features + label
needed_cols = feature_columns + ["Label_Binary"]
cicids_df = cicids_df[needed_cols].copy()

print(f"CICIDS full shape: {cicids_df.shape}")

if USE_CICIDS_SAMPLE and len(cicids_df) > CICIDS_SAMPLE_SIZE:
    print(f"Taking stratified CICIDS sample of {CICIDS_SAMPLE_SIZE:,} rows...")
    _, cicids_df = train_test_split(
        cicids_df,
        test_size=CICIDS_SAMPLE_SIZE,
        stratify=cicids_df["Label_Binary"],
        random_state=RANDOM_STATE,
    )
    print(f"CICIDS sampled shape: {cicids_df.shape}")

print("\nCICIDS label distribution:")
print(cicids_df["Label_Binary"].value_counts())


# =========================
# LOAD CUSTOM DATA
# =========================
print("\nLoading custom dataset...")
custom_df = pd.read_csv(CUSTOM_FILE, low_memory=False)

if "Label_Binary" not in custom_df.columns:
    raise ValueError("Label_Binary column not found in custom file.")

# Custom file has Source_Group and Source_File too; remove them from X
keep_cols = [c for c in feature_columns if c in custom_df.columns] + ["Label_Binary"]
custom_df = custom_df[keep_cols].copy()

print(f"Custom full shape: {custom_df.shape}")
print("\nCustom label distribution before balancing:")
print(custom_df["Label_Binary"].value_counts())

if USE_CUSTOM_BALANCING:
    custom_normal = custom_df[custom_df["Label_Binary"] == 0].copy()
    custom_attack = custom_df[custom_df["Label_Binary"] == 1].copy()

    target_attack_rows = int(len(custom_normal) * CUSTOM_ATTACK_NORMAL_RATIO)
    target_attack_rows = min(target_attack_rows, MAX_CUSTOM_ATTACK_ROWS)

    if target_attack_rows > 0 and len(custom_attack) > target_attack_rows:
        custom_attack = custom_attack.sample(
            n=target_attack_rows,
            random_state=RANDOM_STATE
        )

    custom_df = pd.concat([custom_normal, custom_attack], ignore_index=True)

    print("\nCustom label distribution after balancing:")
    print(custom_df["Label_Binary"].value_counts())

print(f"Custom used shape: {custom_df.shape}")


# =========================
# MERGE DATASETS
# =========================
print("\nMerging CICIDS2017 + custom data...")
full_df = pd.concat([cicids_df, custom_df], ignore_index=True)

before_dups = len(full_df)
full_df.drop_duplicates(inplace=True)
removed_dups = before_dups - len(full_df)

print(f"Duplicate rows removed after merge: {removed_dups}")
print(f"Final merged shape: {full_df.shape}")

print("\nFinal merged label distribution:")
print(full_df["Label_Binary"].value_counts())


# =========================
# PREPARE X / y
# =========================
X = full_df[feature_columns].copy()
y = full_df["Label_Binary"].astype(int)

X = X.apply(pd.to_numeric, errors="coerce")
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0.0, inplace=True)

X = X.astype(np.float32)

print(f"\nNumber of features: {len(feature_columns)}")

# Save feature list for improved model too
pd.DataFrame({"feature_name": feature_columns}).to_csv(
    RESULTS_DIR / "training_feature_columns_improved.csv", index=False
)


# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print(f"\nTrain shape: {X_train.shape}")
print(f"Test shape : {X_test.shape}")


# =========================
# SCALE
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, MODELS_DIR / "scaler_improved.pkl")
print(f"\nSaved improved scaler to: {MODELS_DIR / 'scaler_improved.pkl'}")


# =========================
# TRAIN IMPROVED RANDOM FOREST
# =========================
print("\n=========================")
print("Training Improved Random Forest...")
print("=========================")

rf = RandomForestClassifier(
    n_estimators=150,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced_subsample",
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=3
)

start = time.time()
rf.fit(X_train_scaled, y_train)
train_time = time.time() - start

y_pred = rf.predict(X_test_scaled)

metrics, cm, report = evaluate_model("Improved Random Forest", y_test, y_pred, train_time)

joblib.dump(rf, MODELS_DIR / "random_forest_improved.pkl")
plot_conf_matrix(cm, "Improved Random Forest - Confusion Matrix", RESULTS_DIR / "cm_random_forest_improved.png")

with open(RESULTS_DIR / "report_random_forest_improved.txt", "w", encoding="utf-8") as f:
    f.write(report)

results_df = pd.DataFrame([metrics])
results_df.to_csv(RESULTS_DIR / "model_comparison_results_improved.csv", index=False)

best_info = {
    "best_model": "Improved Random Forest",
    "feature_count": len(feature_columns),
    "use_cicids_sample": USE_CICIDS_SAMPLE,
    "cicids_sample_size": CICIDS_SAMPLE_SIZE if USE_CICIDS_SAMPLE else None,
    "custom_balanced": USE_CUSTOM_BALANCING,
    "custom_attack_normal_ratio": CUSTOM_ATTACK_NORMAL_RATIO,
    "max_custom_attack_rows": MAX_CUSTOM_ATTACK_ROWS,
}

with open(MODELS_DIR / "best_model_info_improved.json", "w", encoding="utf-8") as f:
    json.dump(best_info, f, indent=4)

print("\n=========================")
print("IMPROVED MODEL RESULTS")
print("=========================")
print(results_df)

print(f"\nSaved improved model to: {MODELS_DIR / 'random_forest_improved.pkl'}")
print(f"Saved improved metrics to: {RESULTS_DIR / 'model_comparison_results_improved.csv'}")
print(f"Saved improved report to: {RESULTS_DIR / 'report_random_forest_improved.txt'}")
print(f"Saved improved model info to: {MODELS_DIR / 'best_model_info_improved.json'}")
print("\nDone.")