import pandas as pd
import numpy as np
from pathlib import Path

base = Path(r"C:/Users/g1014308/Desktop")
path = base / "archive_4" / "anilist_anime_data_complete.csv"
print(f"FILE: {path}")

# Load CSV
try:
    df = pd.read_csv(path)
except UnicodeDecodeError:
    df = pd.read_csv(path, encoding='latin1')

print("\n[1] SHAPE")
print(f"rows={df.shape[0]}, cols={df.shape[1]}")

print("\n[2] COLUMNS_AND_DTYPES")
for c, t in df.dtypes.items():
    print(f"{c}\t{t}")

print("\n[3] MISSING_RATE_TOP15")
miss = (df.isna().mean().sort_values(ascending=False) * 100)
for c, v in miss.head(15).items():
    print(f"{c}\t{v:.2f}%")

print("\n[4] TARGET_RELATED_STATS")
target_cols = [c for c in ["score","popularity","favorites","members","ranked","scored_by","num_episodes","duration"] if c in df.columns]
if target_cols:
    desc = df[target_cols].describe(percentiles=[0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99], include='all').T
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(desc.to_string())
else:
    print("No target-related columns found")

print("\n[5] CATEGORICAL_TOP10")
cat_cols = [c for c in ["type","status","source","rating","genres"] if c in df.columns]
if cat_cols:
    for c in cat_cols:
        print(f"-- {c} --")
        vc = df[c].astype('string').value_counts(dropna=False).head(10)
        for k, v in vc.items():
            print(f"{repr(k)}\t{v}")
else:
    print("No requested categorical columns found")

print("\n[6] DATE_RANGE_AND_ANOMALIES")
# season/year raw summary if present
for c in ["season","year"]:
    if c in df.columns:
        ser = df[c]
        print(f"-- {c} --")
        print(f"non_null={ser.notna().sum()}, null={ser.isna().sum()}, unique={ser.nunique(dropna=True)}")
        if pd.api.types.is_numeric_dtype(ser):
            print(f"min={ser.min()}, max={ser.max()}")
        else:
            print(f"top_values={ser.astype('string').value_counts(dropna=False).head(10).to_dict()}")

# parse date columns
for c in ["aired_from","aired_to"]:
    if c in df.columns:
        dt = pd.to_datetime(df[c], errors='coerce', utc=True)
        print(f"-- {c} --")
        print(f"parsed_non_null={dt.notna().sum()}, parse_fail_or_null={(dt.isna()).sum()}")
        if dt.notna().any():
            print(f"min={dt.min()}, max={dt.max()}")

if set(["aired_from","aired_to"]).issubset(df.columns):
    af = pd.to_datetime(df["aired_from"], errors='coerce', utc=True)
    at = pd.to_datetime(df["aired_to"], errors='coerce', utc=True)
    bad_order = ((af.notna()) & (at.notna()) & (at < af)).sum()
    print(f"aired_to_earlier_than_aired_from={int(bad_order)}")

# year anomalies
if "year" in df.columns:
    y = pd.to_numeric(df["year"], errors='coerce')
    anomaly = ((y.notna()) & ((y < 1900) | (y > 2100))).sum()
    print(f"year_out_of_[1900,2100]={int(anomaly)}")

print("\n[7] DUPLICATES_AND_KEY_UNIQUENESS")
dup_rows = df.duplicated().sum()
print(f"duplicate_rows={int(dup_rows)}")
for key in ["id","mal_id","anilist_id"]:
    if key in df.columns:
        nunique = df[key].nunique(dropna=True)
        non_null = df[key].notna().sum()
        dup_non_null = non_null - nunique
        print(f"{key}: non_null={non_null}, unique_non_null={nunique}, duplicated_non_null={dup_non_null}, has_null={df[key].isna().any()}")

print("\n[8] CLEANING_RECOMMENDATIONS_BASIS")
# Basic signals for recommendation support
num_cols = df.select_dtypes(include=[np.number]).columns
if len(num_cols) > 0:
    skew = df[num_cols].skew(numeric_only=True).sort_values(ascending=False)
    print("high_skew_top10:")
    print(skew.head(10).to_string())

