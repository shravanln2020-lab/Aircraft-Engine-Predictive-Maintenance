from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Standard column names for the original NASA C-MAPSS FD001 text files.
RAW_COLUMNS = (
    ["unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3"]
    + [f"s_{i}" for i in range(1, 22)]
)

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 12)
pd.set_option("display.max_colwidth", 24)


def heading(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def summary_table(items):
    table = pd.DataFrame(items, columns=["Dataset file", "Rows", "Columns"])
    print("\nDATASET SUMMARY")
    print(table.to_string(index=False))


def preview(title, df, columns, rows=5):
    print(f"\n{title} — first {min(rows, len(df))} rows")
    available = [column for column in columns if column in df.columns]
    if available:
        print(df[available].head(rows).to_string(index=False))
    else:
        print(df.head(rows).to_string(index=False))


def read_raw(filename):
    path = RAW / filename
    if not path.exists():
        print(f"\nMissing file: {path}")
        return None
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] == len(RAW_COLUMNS):
        df.columns = RAW_COLUMNS
    else:
        df.columns = [f"column_{i + 1}" for i in range(df.shape[1])]
    return df


def read_csv(filename):
    path = PROCESSED / filename
    if not path.exists():
        print(f"\nMissing file: {path}")
        return None
    return pd.read_csv(path)


# -----------------------------------------------------------------------------
# STAGE 1: BEFORE PREPROCESSING — original NASA files
# -----------------------------------------------------------------------------
heading("STAGE 1: BEFORE PREPROCESSING — ORIGINAL NASA C-MAPSS FD001 DATA")
raw_train = read_raw("train_FD001.txt")
raw_test = read_raw("test_FD001.txt")
rul_path = RAW / "RUL_FD001.txt"
raw_rul = pd.read_csv(rul_path, header=None, names=["RUL"]) if rul_path.exists() else None

raw_items = []
for name, df in [("train_FD001.txt", raw_train), ("test_FD001.txt", raw_test), ("RUL_FD001.txt", raw_rul)]:
    if df is not None:
        raw_items.append((name, df.shape[0], df.shape[1]))
summary_table(raw_items)

raw_preview_cols = ["unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3", "s_1", "s_2", "s_3"]
if raw_train is not None:
    preview("Raw training data", raw_train, raw_preview_cols)
if raw_test is not None:
    preview("Raw testing data", raw_test, raw_preview_cols)
if raw_rul is not None:
    preview("RUL reference file", raw_rul, ["RUL"])


# -----------------------------------------------------------------------------
# STAGE 2: AFTER PREPROCESSING
# -----------------------------------------------------------------------------
heading("STAGE 2: AFTER PREPROCESSING")
train_processed = read_csv("train_processed.csv")
test_processed = read_csv("test_processed.csv")
processed_items = []
for name, df in [("train_processed.csv", train_processed), ("test_processed.csv", test_processed)]:
    if df is not None:
        processed_items.append((name, df.shape[0], df.shape[1]))
summary_table(processed_items)
processed_preview_cols = ["unit_nr", "time_cycles", "setting_1", "setting_2", "s_1", "s_2", "RUL"]
if train_processed is not None:
    preview("Preprocessed training data", train_processed, processed_preview_cols)
if test_processed is not None:
    preview("Preprocessed testing data", test_processed, processed_preview_cols)


# -----------------------------------------------------------------------------
# STAGE 3: AFTER FEATURE ENGINEERING
# -----------------------------------------------------------------------------
heading("STAGE 3: AFTER FEATURE ENGINEERING")
train_features = read_csv("train_features.csv")
test_features = read_csv("test_features.csv")
feature_items = []
for name, df in [("train_features.csv", train_features), ("test_features.csv", test_features)]:
    if df is not None:
        feature_items.append((name, df.shape[0], df.shape[1]))
summary_table(feature_items)
feature_preview_cols = [
    "unit_nr", "time_cycles", "RUL", "engine_age", "life_percentage",
    "s_2_delta", "s_2_roll_mean_5", "s_2_roll_std_5", "health_index"
]
if train_features is not None:
    preview("Feature-engineered training data", train_features, feature_preview_cols)
if test_features is not None:
    preview("Feature-engineered testing data", test_features, feature_preview_cols)

print("\n" + "=" * 88)
print("DONE — all available dataset stages have been displayed.")
print("Note: previews show selected columns for readability; summary tables show full dimensions.")
print("=" * 88)
