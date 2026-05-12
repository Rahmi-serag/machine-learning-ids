from pathlib import Path
import time
import shutil
import subprocess
import joblib
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
BASE_DIR = Path(r"C:\Users\rahmy\Desktop\ELTE\UNI 6\Diploma\IDS2\IDS_NORMAL")

LIVEFEED_DIR = Path(r"C:\Users\rahmy\Desktop\ELTE\UNI 6\Diploma\IDS2\LiveFeed")
LIVEFEED_CSV_DIR = Path(r"C:\Users\rahmy\Desktop\ELTE\UNI 6\Diploma\IDS2\LiveFeed_CSV")
TMP_CFM_INPUT_DIR = BASE_DIR / "tmp_cfm_input"

ARCHIVE_DIR = BASE_DIR / "data" / "live_pcap_archive"
FAILED_PCAP_DIR = BASE_DIR / "data" / "live_pcap_failed"

RESULTS_DIR = BASE_DIR / "results" / "live_predictions_improved"
LOGS_DIR = BASE_DIR / "logs"

MODEL_PATH = BASE_DIR / "models" / "random_forest_improved.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler_improved.pkl"
FEATURES_PATH = BASE_DIR / "results" / "training_feature_columns_improved.csv"

CICFLOWMETER_BAT = Path(
    r"C:\Users\rahmy\Desktop\ELTE\UNI 6\Diploma\IDS2\CICFlowMeter-master\build\distributions\CICFlowMeter-4.0\bin\cfm.bat"
)

PROCESSED_PCAP_LOG = LOGS_DIR / "processed_pcap_files_improved.json"
PROCESSED_CSV_LOG = LOGS_DIR / "processed_csv_files_improved.json"
SUMMARY_LOG = LOGS_DIR / "live_detection_summary_improved.csv"

POLL_SECONDS = 3
CSV_WAIT_SECONDS = 2

# PCAP handling
PCAP_STABLE_WAIT_SECONDS = 1.0
PCAP_MIN_AGE_SECONDS = 5.0
EMPTY_PCAP_MAX_BYTES = 24

# Detection logic
ATTACK_RATIO_THRESHOLD = 0.01
MIN_ATTACK_FLOWS = 10

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

DROP_COLUMNS = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"]

# =========================
# SETUP
# =========================
TMP_CFM_INPUT_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
FAILED_PCAP_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LIVEFEED_DIR.mkdir(parents=True, exist_ok=True)
LIVEFEED_CSV_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# HELPERS
# =========================
def load_json_set(path: Path) -> set[str]:
    if path.exists():
        try:
            return set(pd.read_json(path, typ="series").tolist())
        except Exception:
            pass
    return set()


def save_json_set(path: Path, values: set[str]) -> None:
    pd.Series(sorted(values)).to_json(path)


def append_summary_row(row: dict) -> None:
    row_df = pd.DataFrame([row])
    if SUMMARY_LOG.exists():
        row_df.to_csv(SUMMARY_LOG, mode="a", index=False, header=False)
    else:
        row_df.to_csv(SUMMARY_LOG, mode="w", index=False, header=True)


def load_feature_columns() -> list[str]:
    df = pd.read_csv(FEATURES_PATH)
    return df["feature_name"].tolist()


def get_pcap_status(path: Path) -> str:
    """
    Returns:
    - "ready" if the PCAP is stable and large enough to process
    - "empty" if the PCAP is stable but contains no packet data
    - "not_ready" if the file is still being written or copied
    """

    if not path.exists():
        return "not_ready"

    file_age = time.time() - path.stat().st_mtime
    if file_age < PCAP_MIN_AGE_SECONDS:
        return "not_ready"

    size_1 = path.stat().st_size
    time.sleep(PCAP_STABLE_WAIT_SECONDS)

    if not path.exists():
        return "not_ready"

    size_2 = path.stat().st_size

    if size_1 != size_2:
        return "not_ready"

    # A PCAP file with only the global header is usually 24 bytes.
    # That means tcpdump created a file but captured no packets.
    if size_2 <= EMPTY_PCAP_MAX_BYTES:
        return "empty"

    return "ready"


def delete_empty_pcap(pcap_path: Path):
    print(f"Deleting empty PCAP: {pcap_path.name}")

    try:
        if pcap_path.exists():
            pcap_path.unlink()
    except Exception as e:
        print(f"Could not delete empty PCAP {pcap_path.name}: {e}")
        return

    append_summary_row({
        "pcap_or_csv_file": pcap_path.name,
        "total_flows": 0,
        "normal_flows": 0,
        "attack_flows": 0,
        "attack_ratio": 0.0,
        "decision": "EMPTY_PCAP_DELETED",
        "model_used": "random_forest_improved.pkl"
    })


def get_new_pcaps(processed_pcaps: set[str]) -> list[Path]:
    ready_pcaps = []

    pcaps = sorted(
        [p for p in LIVEFEED_DIR.glob("*.pcap") if p.name not in processed_pcaps]
    )

    for pcap in pcaps:
        status = get_pcap_status(pcap)

        if status == "ready":
            ready_pcaps.append(pcap)

        elif status == "empty":
            delete_empty_pcap(pcap)
            processed_pcaps.add(pcap.name)
            save_json_set(PROCESSED_PCAP_LOG, processed_pcaps)

        else:
            # File is still being written or copied.
            # Wait silently and check again later.
            pass

    return ready_pcaps


def clean_temp_input_dir() -> None:
    for old in TMP_CFM_INPUT_DIR.glob("*"):
        try:
            if old.is_file():
                old.unlink()
        except Exception:
            pass


def convert_pcap_with_cicflowmeter(pcap_path: Path) -> list[Path]:
    print(f"\nConverting PCAP with CICFlowMeter: {pcap_path.name}")

    clean_temp_input_dir()

    temp_pcap = TMP_CFM_INPUT_DIR / pcap_path.name
    shutil.copy2(pcap_path, temp_pcap)

    expected_csv = LIVEFEED_CSV_DIR / f"{pcap_path.name}_Flow.csv"

    # Remove old CSV with the same name so the script can detect fresh output.
    if expected_csv.exists():
        try:
            expected_csv.unlink()
        except Exception:
            pass

    before = {p.name for p in LIVEFEED_CSV_DIR.glob("*.csv")}
    cfm_dir = CICFLOWMETER_BAT.parent

    command = f'"{CICFLOWMETER_BAT}" "{TMP_CFM_INPUT_DIR}" "{LIVEFEED_CSV_DIR}"'

    result = subprocess.run(
        command,
        cwd=str(cfm_dir),
        capture_output=True,
        text=True,
        shell=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"CICFlowMeter failed for {pcap_path.name}\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

    time.sleep(CSV_WAIT_SECONDS)

    if expected_csv.exists() and expected_csv.stat().st_size > 0:
        new_csv_files = [expected_csv]
    else:
        after = {p.name for p in LIVEFEED_CSV_DIR.glob("*.csv")}
        new_csv_names = sorted(after - before)
        new_csv_files = [LIVEFEED_CSV_DIR / name for name in new_csv_names]

    if not new_csv_files:
        raise RuntimeError(f"No CSV file was generated for {pcap_path.name}")

    print("Generated CSV file(s):")
    for csv_file in new_csv_files:
        print(f" - {csv_file.name}")

    return new_csv_files


def prepare_csv_for_model(csv_path: Path, feature_columns: list[str]):
    df = pd.read_csv(csv_path, low_memory=False)

    if df.empty:
        raise ValueError("CSV file is empty.")

    df.columns = df.columns.str.strip()
    df.rename(columns=COLUMN_RENAME_MAP, inplace=True)

    if "Fwd Header Length.1" in df.columns:
        df.drop(columns=["Fwd Header Length.1"], inplace=True)

    meta_cols = [
        c for c in [
            "Flow ID",
            "Src IP",
            "Src Port",
            "Dst IP",
            "Dst Port",
            "Protocol",
            "Timestamp"
        ]
        if c in df.columns
    ]

    meta_df = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)

    for col in DROP_COLUMNS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    missing_cols = [c for c in feature_columns if c not in df.columns]
    for col in missing_cols:
        df[col] = 0.0

    df = df[feature_columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)
    df = df.astype(np.float32)

    return meta_df, df


def classify_csv(csv_path: Path, model, scaler, feature_columns: list[str]) -> dict:
    print(f"\nClassifying CSV: {csv_path.name}")

    try:
        meta_df, X = prepare_csv_for_model(csv_path, feature_columns)
    except ValueError as e:
        if "CSV file is empty" in str(e):
            print("CSV is empty. Marking as NO_DATA and continuing.")

            append_summary_row({
                "pcap_or_csv_file": csv_path.name,
                "total_flows": 0,
                "normal_flows": 0,
                "attack_flows": 0,
                "attack_ratio": 0.0,
                "decision": "NO_DATA",
                "model_used": "random_forest_improved.pkl"
            })

            return {
                "csv_name": csv_path.name,
                "total_flows": 0,
                "normal_flows": 0,
                "attack_flows": 0,
                "attack_ratio": 0.0,
                "decision": "NO_DATA",
                "output_file": None,
            }

        raise

    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]

    result_df = meta_df.copy()
    result_df["Predicted_Label_Binary"] = preds
    result_df["Attack_Probability"] = probs

    total_flows = int(len(result_df))
    attack_flows = int((result_df["Predicted_Label_Binary"] == 1).sum())
    normal_flows = total_flows - attack_flows
    attack_ratio = (attack_flows / total_flows) if total_flows > 0 else 0.0

    decision = (
        "ATTACK"
        if (attack_flows >= MIN_ATTACK_FLOWS and attack_ratio >= ATTACK_RATIO_THRESHOLD)
        else "NORMAL"
    )

    output_file = RESULTS_DIR / csv_path.name.replace(".csv", "_predictions.csv")
    result_df.to_csv(output_file, index=False)

    print(f"Total flows  : {total_flows}")
    print(f"Normal flows : {normal_flows}")
    print(f"Attack flows : {attack_flows}")
    print(f"Attack ratio : {attack_ratio:.4f}")
    print(f"Decision     : {decision}")
    print(f"Saved        : {output_file}")

    append_summary_row({
        "pcap_or_csv_file": csv_path.name,
        "total_flows": total_flows,
        "normal_flows": normal_flows,
        "attack_flows": attack_flows,
        "attack_ratio": round(attack_ratio, 6),
        "decision": decision,
        "model_used": "random_forest_improved.pkl"
    })

    return {
        "csv_name": csv_path.name,
        "total_flows": total_flows,
        "normal_flows": normal_flows,
        "attack_flows": attack_flows,
        "attack_ratio": attack_ratio,
        "decision": decision,
        "output_file": str(output_file),
    }


def archive_pcap(pcap_path: Path):
    target = ARCHIVE_DIR / pcap_path.name

    if target.exists():
        target.unlink()

    shutil.move(str(pcap_path), str(target))
    print(f"Archived PCAP to: {target}")


def archive_failed_pcap(pcap_path: Path, reason: str):
    target = FAILED_PCAP_DIR / pcap_path.name

    if target.exists():
        target.unlink()

    if pcap_path.exists():
        shutil.move(str(pcap_path), str(target))

    append_summary_row({
        "pcap_or_csv_file": pcap_path.name,
        "total_flows": 0,
        "normal_flows": 0,
        "attack_flows": 0,
        "attack_ratio": 0.0,
        "decision": "NO_CSV",
        "model_used": "random_forest_improved.pkl"
    })

    print(f"Moved failed PCAP to: {target}")
    print(f"Reason: {reason}")


print("Loading improved model, scaler, and feature list...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = load_feature_columns()

processed_pcaps = load_json_set(PROCESSED_PCAP_LOG)
processed_csvs = load_json_set(PROCESSED_CSV_LOG)

print(f"Watching PCAP folder: {LIVEFEED_DIR}")
print(f"CSV output folder   : {LIVEFEED_CSV_DIR}")
print(f"Archive folder      : {ARCHIVE_DIR}")
print(f"Failed PCAP folder  : {FAILED_PCAP_DIR}")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        new_pcaps = get_new_pcaps(processed_pcaps)

        for pcap_path in new_pcaps:
            try:
                new_csv_files = convert_pcap_with_cicflowmeter(pcap_path)

                for csv_file in new_csv_files:
                    if csv_file.name not in processed_csvs:
                        classify_csv(csv_file, model, scaler, feature_columns)
                        processed_csvs.add(csv_file.name)
                        save_json_set(PROCESSED_CSV_LOG, processed_csvs)

                archive_pcap(pcap_path)

                processed_pcaps.add(pcap_path.name)
                save_json_set(PROCESSED_PCAP_LOG, processed_pcaps)

            except Exception as e:
                error_message = str(e)
                print(f"\nError handling {pcap_path.name}: {error_message}\n")

                
                if "No CSV file was generated" in error_message:
                    archive_failed_pcap(pcap_path, error_message)
                    processed_pcaps.add(pcap_path.name)
                    save_json_set(PROCESSED_PCAP_LOG, processed_pcaps)


        time.sleep(POLL_SECONDS)

except KeyboardInterrupt:
    print("\nStopped by user.")