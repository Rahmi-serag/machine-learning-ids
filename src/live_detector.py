from pathlib import Path
import time
import json
import joblib
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
BASE_DIR = Path(r"C:\Users\rahmy\Desktop\ELTE\UNI 6\Diploma\IDS2\IDS_NORMAL")
LIVE_CSV_DIR = Path(r"C:\Users\rahmy\Desktop\ELTE\UNI 6\Diploma\IDS2\LiveFeed_CSV")

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"
LIVE_RESULTS_DIR = RESULTS_DIR / "live_predictions"

LIVE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "random_forest.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURES_PATH = RESULTS_DIR / "training_feature_columns.csv"

PROCESSED_FILES_LOG = LOGS_DIR / "processed_live_files.json"
SUMMARY_LOG = LOGS_DIR / "live_detection_summary.csv"

POLL_INTERVAL_SECONDS = 5
FILE_READY_AGE_SECONDS = 2

# Overall file decision rule
MIN_ATTACK_FLOWS = 3
ATTACK_RATIO_THRESHOLD = 0.02   # 2%

ENCODINGS_TO_TRY = ["utf-8", "cp1252", "latin1"]

# Rename old / alternate names to the model's expected names
COLUMN_RENAME_MAP = {
    "Source IP": "Src IP",
    "Source Port": "Src Port",
    "Destination IP": "Dst IP",
    "Destination Port": "Dst Port",
    "Total Fwd Packets": "Total Fwd Packet",
    "Total Backward Packets": "Total Bwd packets",
    "Total Length of Fwd Packets": "Total Length of Fwd Packet",
    "Total Length of Bwd Packets": "Total Length of Bwd Packet",
    "Min Packet Length": "Packet Length Min",
    "Max Packet Length": "Packet Length Max",
    "CWE Flag Count": "CWR Flag Count",
    "Avg Fwd Segment Size": "Fwd Segment Size Avg",
    "Avg Bwd Segment Size": "Bwd Segment Size Avg",
    "Fwd Avg Bytes/Bulk": "Fwd Bytes/Bulk Avg",
    "Fwd Avg Packets/Bulk": "Fwd Packet/Bulk Avg",
    "Fwd Avg Bulk Rate": "Fwd Bulk Rate Avg",
    "Bwd Avg Bytes/Bulk": "Bwd Bytes/Bulk Avg",
    "Bwd Avg Packets/Bulk": "Bwd Packet/Bulk Avg",
    "Bwd Avg Bulk Rate": "Bwd Bulk Rate Avg",
    "Init_Win_bytes_forward": "FWD Init Win Bytes",
    "Init_Win_bytes_backward": "Bwd Init Win Bytes",
    "act_data_pkt_fwd": "Fwd Act Data Pkts",
    "min_seg_size_forward": "Fwd Seg Size Min",
}

DROP_COLUMNS = [
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp",
    "Label",
]

# =========================
# HELPERS
# =========================
def load_feature_columns():
    df = pd.read_csv(FEATURES_PATH)
    return df["feature_name"].tolist()


def load_processed_files():
    if PROCESSED_FILES_LOG.exists():
        try:
            return set(json.loads(PROCESSED_FILES_LOG.read_text(encoding="utf-8")))
        except Exception:
            return set()
    return set()


def save_processed_files(processed_files):
    PROCESSED_FILES_LOG.write_text(
        json.dumps(sorted(list(processed_files)), indent=2),
        encoding="utf-8"
    )


def append_summary_row(row_dict):
    df_row = pd.DataFrame([row_dict])
    if SUMMARY_LOG.exists():
        df_row.to_csv(SUMMARY_LOG, mode="a", header=False, index=False)
    else:
        df_row.to_csv(SUMMARY_LOG, index=False)


def read_csv_robust(file_path: Path):
    last_error = None
    for enc in ENCODINGS_TO_TRY:
        try:
            df = pd.read_csv(file_path, encoding=enc, low_memory=False, on_bad_lines="skip")
            return df
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Could not read {file_path.name}. Last error: {last_error}")


def prepare_live_dataframe(df, feature_columns):
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Unify names
    df.rename(columns=COLUMN_RENAME_MAP, inplace=True)

    # Drop duplicate style column if present
    if "Fwd Header Length.1" in df.columns:
        df.drop(columns=["Fwd Header Length.1"], inplace=True)

    # Keep a copy of some optional metadata columns for output
    metadata_cols = [c for c in ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp"] if c in df.columns]
    meta_df = df[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=df.index)

    # Drop non-feature columns
    for col in DROP_COLUMNS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Add any missing expected feature columns with 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0

    # Keep only expected features and in the exact same order
    df = df[feature_columns]

    # Convert all to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Replace inf with NaN, then fill with 0 for live robustness
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    nan_count = int(df.isna().sum().sum())
    if nan_count > 0:
        df.fillna(0.0, inplace=True)

    return meta_df, df, nan_count


def predict_file(file_path: Path, model, scaler, feature_columns):
    print(f"\nProcessing: {file_path.name}")

    raw_df = read_csv_robust(file_path)

    if raw_df.empty:
        raise ValueError("CSV file is empty.")

    meta_df, X_live, nan_count = prepare_live_dataframe(raw_df, feature_columns)

    if X_live.empty:
        raise ValueError("No usable rows after preprocessing.")

    X_scaled = scaler.transform(X_live)
    preds = model.predict(X_scaled)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[:, 1]
    else:
        probs = np.full(len(preds), np.nan)

    attack_count = int((preds == 1).sum())
    normal_count = int((preds == 0).sum())
    total_flows = int(len(preds))
    attack_ratio = attack_count / total_flows if total_flows > 0 else 0.0

    overall_decision = "ATTACK" if (
        attack_count >= MIN_ATTACK_FLOWS and attack_ratio >= ATTACK_RATIO_THRESHOLD
    ) else "NORMAL"

    output_df = meta_df.copy()
    output_df["Predicted_Label_Binary"] = preds
    output_df["Predicted_Label_Text"] = np.where(preds == 1, "ATTACK", "NORMAL")
    output_df["Attack_Probability"] = probs

    output_path = LIVE_RESULTS_DIR / f"{file_path.stem}_predictions.csv"
    output_df.to_csv(output_path, index=False)

    summary = {
        "file_name": file_path.name,
        "total_flows": total_flows,
        "normal_flows": normal_count,
        "attack_flows": attack_count,
        "attack_ratio": round(attack_ratio, 6),
        "overall_decision": overall_decision,
        "nan_cells_filled_with_zero": nan_count,
        "prediction_file": str(output_path),
        "processed_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    append_summary_row(summary)

    print(f"Total flows   : {total_flows}")
    print(f"Normal flows  : {normal_count}")
    print(f"Attack flows  : {attack_count}")
    print(f"Attack ratio  : {attack_ratio:.4f}")
    print(f"Decision      : {overall_decision}")
    print(f"Saved details : {output_path}")

    return summary


def is_file_ready(file_path: Path):
    age = time.time() - file_path.stat().st_mtime
    return age >= FILE_READY_AGE_SECONDS


def main():
    print("Loading model and scaler...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = load_feature_columns()

    processed_files = load_processed_files()

    print(f"Watching folder: {LIVE_CSV_DIR}")
    print("Press Ctrl+C to stop.\n")

    while True:
        try:
            csv_files = sorted(LIVE_CSV_DIR.glob("*.csv"))

            # avoid processing prediction outputs if user ever copies them here by mistake
            csv_files = [f for f in csv_files if not f.name.endswith("_predictions.csv")]

            for file_path in csv_files:
                file_key = str(file_path.resolve())

                if file_key in processed_files:
                    continue

                if not is_file_ready(file_path):
                    continue

                try:
                    predict_file(file_path, model, scaler, feature_columns)
                    processed_files.add(file_key)
                    save_processed_files(processed_files)
                except Exception as e:
                    print(f"Error processing {file_path.name}: {e}")

            time.sleep(POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("\nStopped by user.")
            break


if __name__ == "__main__":
    main()