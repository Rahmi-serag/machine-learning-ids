from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(r"C:\Users\rahmy\Desktop\ELTE\UNI 6\Diploma\IDS2\IDS_NORMAL")
RAW_DIR = BASE_DIR / "data" / "raw_cicids2017"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Rename CICIDS2017 columns to match your live CICFlowMeter naming as much as possible
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

# Columns to drop because they are identifiers / text-like, not good ML features
DROP_COLUMNS = [
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp",
]

ENCODINGS_TO_TRY = ["utf-8", "cp1252", "latin1"]


def read_csv_robust(file_path: Path) -> pd.DataFrame:
    last_error = None
    for enc in ENCODINGS_TO_TRY:
        try:
            print(f"   Trying encoding: {enc}")
            df = pd.read_csv(
                file_path,
                encoding=enc,
                low_memory=False,
                on_bad_lines="skip"
            )
            print(f"   Success with encoding: {enc}")
            return df
        except UnicodeDecodeError as e:
            last_error = e
            print(f"   Failed with encoding: {enc}")

    raise UnicodeDecodeError(
        "unknown", b"", 0, 1,
        f"Could not read {file_path.name} with tried encodings. Last error: {last_error}"
    )


def main():
    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {RAW_DIR}")

    print(f"Found {len(csv_files)} CSV files.\n")

    dataframes = []

    for i, file in enumerate(csv_files, start=1):
        print(f"[{i}/{len(csv_files)}] Reading: {file.name}")
        df = read_csv_robust(file)
        df.columns = df.columns.str.strip()
        dataframes.append(df)

    print("\nMerging all files...")
    df_all = pd.concat(dataframes, ignore_index=True)
    df_all.columns = df_all.columns.str.strip()

    print(f"Shape after merge: {df_all.shape}")

    # Rename columns toward live CICFlowMeter style
    df_all.rename(columns=COLUMN_RENAME_MAP, inplace=True)

    # Drop duplicate/extra column if present
    if "Fwd Header Length.1" in df_all.columns:
        df_all.drop(columns=["Fwd Header Length.1"], inplace=True)
        print("Dropped duplicate column: Fwd Header Length.1")

    if "Label" not in df_all.columns:
        raise ValueError("Column 'Label' not found after processing.")

    # Clean labels
    df_all["Label"] = df_all["Label"].astype(str).str.strip()

    # Save original label summary
    original_label_summary = (
        df_all["Label"]
        .value_counts(dropna=False)
        .rename_axis("Original_Label")
        .reset_index(name="Count")
    )
    original_label_summary.to_csv(PROCESSED_DIR / "original_label_summary.csv", index=False)

    # Binary label: BENIGN=0, anything else=1
    df_all["Label_Binary"] = (df_all["Label"].str.upper() != "BENIGN").astype(int)

    print("\nBinary label distribution before cleaning:")
    print(df_all["Label_Binary"].value_counts())

    # Drop non-ML identifier columns
    existing_drop_cols = [col for col in DROP_COLUMNS if col in df_all.columns]
    if existing_drop_cols:
        df_all.drop(columns=existing_drop_cols, inplace=True)
        print(f"\nDropped identifier columns: {existing_drop_cols}")

    # Convert all remaining feature columns to numeric
    feature_cols = [col for col in df_all.columns if col not in ["Label", "Label_Binary"]]
    df_all[feature_cols] = df_all[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Replace inf with NaN
    df_all.replace([np.inf, -np.inf], np.nan, inplace=True)

    before_dropna = len(df_all)
    df_all.dropna(inplace=True)
    after_dropna = len(df_all)
    print(f"Rows removed because of NaN/Inf: {before_dropna - after_dropna}")

    before_duplicates = len(df_all)
    df_all.drop_duplicates(inplace=True)
    after_duplicates = len(df_all)
    print(f"Duplicate rows removed: {before_duplicates - after_duplicates}")

    df_all.reset_index(drop=True, inplace=True)

    print(f"\nFinal shape: {df_all.shape}")
    print("\nBinary label distribution after cleaning:")
    print(df_all["Label_Binary"].value_counts())

    # Save processed dataset
    output_file = PROCESSED_DIR / "cicids2017_binary.csv"
    df_all.to_csv(output_file, index=False)

    # Save binary summary
    binary_summary = (
        df_all["Label_Binary"]
        .value_counts()
        .rename_axis("Label_Binary")
        .reset_index(name="Count")
    )
    binary_summary.to_csv(PROCESSED_DIR / "binary_label_summary.csv", index=False)

    # Save feature list for later live detection
    feature_list = pd.DataFrame({
        "feature_name": [col for col in df_all.columns if col not in ["Label", "Label_Binary"]]
    })
    feature_list.to_csv(PROCESSED_DIR / "feature_columns.csv", index=False)

    print(f"\nSaved processed dataset to: {output_file}")
    print(f"Saved original label summary to: {PROCESSED_DIR / 'original_label_summary.csv'}")
    print(f"Saved binary label summary to: {PROCESSED_DIR / 'binary_label_summary.csv'}")
    print(f"Saved feature columns to: {PROCESSED_DIR / 'feature_columns.csv'}")
    print("\nDone.")


if __name__ == "__main__":
    main()