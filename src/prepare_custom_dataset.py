from pathlib import Path
import pandas as pd
import numpy as np

# =========================
# PATHS
# =========================
BASE_DIR = Path(r"C:\Users\rahmy\Desktop\ELTE\UNI 6\Diploma\IDS2\IDS_NORMAL")

NORMAL_LAB_DIR = BASE_DIR / "custom_data" / "csv" / "normal_lab"
NORMAL_NAT_DIR = BASE_DIR / "custom_data" / "csv" / "normal_nat"
ATTACK_DIR = BASE_DIR / "custom_data" / "csv" / "attack"
OUTPUT_DIR = BASE_DIR / "custom_data" / "processed"

FEATURES_FILE_1 = BASE_DIR / "results" / "training_feature_columns.csv"
FEATURES_FILE_2 = BASE_DIR / "data" / "processed" / "feature_columns.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DATASET = OUTPUT_DIR / "custom_binary_dataset.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "custom_dataset_summary.csv"

ENCODINGS_TO_TRY = ["utf-8", "cp1252", "latin1"]

# =========================
# COLUMN NAME ALIGNMENT
# =========================
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
def load_feature_columns() -> list[str]:
    if FEATURES_FILE_1.exists():
        df = pd.read_csv(FEATURES_FILE_1)
        return df["feature_name"].tolist()

    if FEATURES_FILE_2.exists():
        df = pd.read_csv(FEATURES_FILE_2)
        first_col = df.columns[0]
        return df[first_col].tolist()

    raise FileNotFoundError(
        "Could not find training feature columns file.\n"
        f"Tried:\n - {FEATURES_FILE_1}\n - {FEATURES_FILE_2}"
    )

def read_csv_robust(file_path: Path) -> pd.DataFrame:
    last_error = None
    for enc in ENCODINGS_TO_TRY:
        try:
            return pd.read_csv(file_path, encoding=enc, low_memory=False, on_bad_lines="skip")
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Could not read {file_path.name}. Last error: {last_error}")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df.rename(columns=COLUMN_RENAME_MAP, inplace=True)

    if "Fwd Header Length.1" in df.columns:
        df.drop(columns=["Fwd Header Length.1"], inplace=True)

    return df

def prepare_one_file(file_path: Path, label_binary: int, source_group: str, feature_columns: list[str]) -> tuple[pd.DataFrame, dict]:
    raw_df = read_csv_robust(file_path)

    if raw_df.empty:
        raise ValueError("CSV file is empty.")

    df = normalize_columns(raw_df)

    for col in DROP_COLUMNS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    missing_cols = [col for col in feature_columns if col not in df.columns]
    for col in missing_cols:
        df[col] = 0.0

    df = df[feature_columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    nan_cells_before_fill = int(df.isna().sum().sum())
    if nan_cells_before_fill > 0:
        df.fillna(0.0, inplace=True)

    df["Label_Binary"] = label_binary
    df["Source_Group"] = source_group
    df["Source_File"] = file_path.name

    summary = {
        "file_name": file_path.name,
        "source_group": source_group,
        "label_binary": label_binary,
        "rows": len(df),
        "missing_columns_added": len(missing_cols),
        "nan_cells_filled": nan_cells_before_fill,
    }

    return df, summary

def collect_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob("*.csv"))

# =========================
# MAIN
# =========================
def main():
    feature_columns = load_feature_columns()
    print(f"Loaded {len(feature_columns)} training feature columns.")

    normal_lab_files = collect_files(NORMAL_LAB_DIR)
    normal_nat_files = collect_files(NORMAL_NAT_DIR)
    attack_files = collect_files(ATTACK_DIR)

    print(f"Normal LAB CSV files found : {len(normal_lab_files)}")
    print(f"Normal NAT CSV files found : {len(normal_nat_files)}")
    print(f"Attack CSV files found     : {len(attack_files)}")

    if not normal_lab_files and not normal_nat_files and not attack_files:
        raise RuntimeError("No CSV files found in custom normal_lab / normal_nat / attack folders.")

    all_parts = []
    summary_rows = []

    # normal_lab
    for i, file_path in enumerate(normal_lab_files, start=1):
        print(f"\n[Normal LAB {i}/{len(normal_lab_files)}] Reading: {file_path.name}")
        try:
            df_part, info = prepare_one_file(file_path, 0, "normal_lab", feature_columns)
            all_parts.append(df_part)
            summary_rows.append(info)
            print(f"   Rows kept: {info['rows']}")
            print(f"   Missing columns added: {info['missing_columns_added']}")
            print(f"   NaN cells filled: {info['nan_cells_filled']}")
        except Exception as e:
            print(f"   Skipped because of error: {e}")

    # normal_nat
    for i, file_path in enumerate(normal_nat_files, start=1):
        print(f"\n[Normal NAT {i}/{len(normal_nat_files)}] Reading: {file_path.name}")
        try:
            df_part, info = prepare_one_file(file_path, 0, "normal_nat", feature_columns)
            all_parts.append(df_part)
            summary_rows.append(info)
            print(f"   Rows kept: {info['rows']}")
            print(f"   Missing columns added: {info['missing_columns_added']}")
            print(f"   NaN cells filled: {info['nan_cells_filled']}")
        except Exception as e:
            print(f"   Skipped because of error: {e}")

    # attack
    for i, file_path in enumerate(attack_files, start=1):
        print(f"\n[Attack {i}/{len(attack_files)}] Reading: {file_path.name}")
        try:
            df_part, info = prepare_one_file(file_path, 1, "attack", feature_columns)
            all_parts.append(df_part)
            summary_rows.append(info)
            print(f"   Rows kept: {info['rows']}")
            print(f"   Missing columns added: {info['missing_columns_added']}")
            print(f"   NaN cells filled: {info['nan_cells_filled']}")
        except Exception as e:
            print(f"   Skipped because of error: {e}")

    if not all_parts:
        raise RuntimeError("No valid custom CSV files were processed.")

    print("\nMerging all custom files...")
    final_df = pd.concat(all_parts, ignore_index=True)

    before_dups = len(final_df)
    final_df.drop_duplicates(inplace=True)
    removed_dups = before_dups - len(final_df)

    print(f"Duplicate rows removed: {removed_dups}")
    print(f"Final custom dataset shape: {final_df.shape}")

    print("\nCustom binary label distribution:")
    print(final_df["Label_Binary"].value_counts())

    print("\nSource group distribution:")
    print(final_df["Source_Group"].value_counts())

    final_df.to_csv(OUTPUT_DATASET, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)

    print(f"\nSaved custom dataset to:\n{OUTPUT_DATASET}")
    print(f"Saved summary to:\n{OUTPUT_SUMMARY}")
    print("\nDone.")

if __name__ == "__main__":
    main()