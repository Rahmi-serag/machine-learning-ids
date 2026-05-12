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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.neural_network import MLPClassifier


# =========================
# CONFIG
# =========================
BASE_DIR = Path(r"C:\Users\rahmy\Desktop\ELTE\UNI 6\Diploma\IDS2\IDS_NORMAL")
DATA_FILE = BASE_DIR / "data" / "processed" / "cicids2017_binary.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Start with a manageable sample on Windows
USE_SAMPLE = True
SAMPLE_SIZE = 200_000

# Neural network settings
NN_EPOCHS = 20
NN_BATCH_SIZE = 1024


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




# =========================
# LOAD DATA
# =========================
print(f"Loading dataset from:\n{DATA_FILE}\n")
df = pd.read_csv(DATA_FILE, low_memory=False)

print(f"Full dataset shape: {df.shape}")

if "Label_Binary" not in df.columns:
    raise ValueError("Label_Binary column not found.")

feature_columns = [col for col in df.columns if col not in ["Label", "Label_Binary"]]

if USE_SAMPLE and len(df) > SAMPLE_SIZE:
    print(f"\nTaking stratified sample of {SAMPLE_SIZE:,} rows...")
    _, df = train_test_split(
        df,
        test_size=SAMPLE_SIZE,
        stratify=df["Label_Binary"],
        random_state=RANDOM_STATE
    )
    print(f"Sampled dataset shape: {df.shape}")

X = df[feature_columns].astype(np.float32)
y = df["Label_Binary"].astype(int)

print(f"\nNumber of features: {len(feature_columns)}")
print("Class distribution:")
print(y.value_counts())

# Save feature names
pd.DataFrame({"feature_name": feature_columns}).to_csv(
    RESULTS_DIR / "training_feature_columns.csv", index=False
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print(f"\nTrain shape: {X_train.shape}")
print(f"Test shape : {X_test.shape}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
print(f"\nSaved scaler to: {MODELS_DIR / 'scaler.pkl'}")

# Class weights
classes = np.array([0, 1])
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weight_dict = {0: float(weights[0]), 1: float(weights[1])}
print(f"\nClass weights: {class_weight_dict}")

results = []

# =========================
# 1) Decision Tree
# =========================
print("\n=========================")
print("Training Decision Tree...")
print("=========================")

dt = DecisionTreeClassifier(
    random_state=RANDOM_STATE,
    class_weight="balanced",
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5
)

start = time.time()
dt.fit(X_train_scaled, y_train)
dt_train_time = time.time() - start

y_pred_dt = dt.predict(X_test_scaled)
metrics_dt, cm_dt, report_dt = evaluate_model("Decision Tree", y_test, y_pred_dt, dt_train_time)
results.append(metrics_dt)

joblib.dump(dt, MODELS_DIR / "decision_tree.pkl")
plot_conf_matrix(cm_dt, "Decision Tree - Confusion Matrix", RESULTS_DIR / "cm_decision_tree.png")

with open(RESULTS_DIR / "report_decision_tree.txt", "w", encoding="utf-8") as f:
    f.write(report_dt)

print(metrics_dt)

# =========================
# 2) Random Forest
# =========================
print("\n=========================")
print("Training Random Forest...")
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
rf_train_time = time.time() - start

y_pred_rf = rf.predict(X_test_scaled)
metrics_rf, cm_rf, report_rf = evaluate_model("Random Forest", y_test, y_pred_rf, rf_train_time)
results.append(metrics_rf)

joblib.dump(rf, MODELS_DIR / "random_forest.pkl")
plot_conf_matrix(cm_rf, "Random Forest - Confusion Matrix", RESULTS_DIR / "cm_random_forest.png")

with open(RESULTS_DIR / "report_random_forest.txt", "w", encoding="utf-8") as f:
    f.write(report_rf)

print(metrics_rf)

# =========================
# 3) SVM
# =========================
print("\n=========================")
print("Training SVM (LinearSVC)...")
print("=========================")

svm_model = LinearSVC(
    class_weight="balanced",
    random_state=RANDOM_STATE,
    max_iter=5000
)

start = time.time()
svm_model.fit(X_train_scaled, y_train)
svm_train_time = time.time() - start

y_pred_svm = svm_model.predict(X_test_scaled)
metrics_svm, cm_svm, report_svm = evaluate_model("SVM", y_test, y_pred_svm, svm_train_time)
results.append(metrics_svm)

joblib.dump(svm_model, MODELS_DIR / "svm.pkl")
plot_conf_matrix(cm_svm, "SVM - Confusion Matrix", RESULTS_DIR / "cm_svm.png")

with open(RESULTS_DIR / "report_svm.txt", "w", encoding="utf-8") as f:
    f.write(report_svm)

print(metrics_svm)

# =========================
# 4) Neural Network (MLPClassifier)
# =========================
print("\n=========================")
print("Training Neural Network (MLPClassifier)...")
print("=========================")

nn_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=1024,
    learning_rate_init=0.001,
    max_iter=30,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=5,
    random_state=RANDOM_STATE,
    verbose=True
)

start = time.time()
nn_model.fit(X_train_scaled, y_train)
nn_train_time = time.time() - start

y_pred_nn = nn_model.predict(X_test_scaled)
metrics_nn, cm_nn, report_nn = evaluate_model("Neural Network", y_test, y_pred_nn, nn_train_time)
results.append(metrics_nn)

joblib.dump(nn_model, MODELS_DIR / "neural_network.pkl")
plot_conf_matrix(cm_nn, "Neural Network - Confusion Matrix", RESULTS_DIR / "cm_neural_network.png")

with open(RESULTS_DIR / "report_neural_network.txt", "w", encoding="utf-8") as f:
    f.write(report_nn)

print(metrics_nn)

# =========================
# SAVE COMPARISON
# =========================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1_Score", ascending=False).reset_index(drop=True)

results_csv = RESULTS_DIR / "model_comparison_results.csv"
results_df.to_csv(results_csv, index=False)

print("\n=========================")
print("FINAL RESULTS")
print("=========================")
print(results_df)

best_model_name = results_df.iloc[0]["Model"]

best_info = {
    "best_model": best_model_name,
    "feature_count": len(feature_columns),
    "sample_used": USE_SAMPLE,
    "sample_size": SAMPLE_SIZE if USE_SAMPLE else None
}

with open(MODELS_DIR / "best_model_info.json", "w", encoding="utf-8") as f:
    json.dump(best_info, f, indent=4)

print(f"\nSaved model comparison to: {results_csv}")
print(f"Saved best model info to: {MODELS_DIR / 'best_model_info.json'}")
print("\nDone.")