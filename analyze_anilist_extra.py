import pandas as pd
import numpy as np
from pathlib import Path

path = Path(r"C:/Users/g1014308/Desktop/archive_4/anilist_anime_data_complete.csv")
try:
    df = pd.read_csv(path)
except UnicodeDecodeError:
    df = pd.read_csv(path, encoding='latin1')

print('[4B] EXTENDED_TARGET_STATS')
# map likely target fields across schema variants
candidate_targets = [
    'score','averageScore','meanScore','popularity','favorites','favourites','members','ranked','rankings','scored_by','episodes','duration','trending'
]
num_targets = [c for c in candidate_targets if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
print('numeric_target_cols=', num_targets)
if num_targets:
    q = [0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]
    out = df[num_targets].describe(percentiles=q).T
    print(out.to_string())

print('\n[6B] DATE_COMPONENT_RANGE_AND_ANOMALY_CHECK')
for c in ['startDate_year','startDate_month','startDate_day','endDate_year','endDate_month','endDate_day','seasonYear','updatedAt']:
    if c in df.columns:
        s = pd.to_numeric(df[c], errors='coerce')
        print(f"{c}: non_null={int(s.notna().sum())}, null={int(s.isna().sum())}, min={s.min()}, max={s.max()}")

# component anomalies
for c, lo, hi in [('startDate_month',1,12),('endDate_month',1,12),('startDate_day',1,31),('endDate_day',1,31)]:
    if c in df.columns:
        s = pd.to_numeric(df[c], errors='coerce')
        bad = ((s.notna()) & ((s < lo) | (s > hi))).sum()
        print(f"{c}_outside_[{lo},{hi}]={int(bad)}")

for c in ['startDate_year','endDate_year','seasonYear']:
    if c in df.columns:
        s = pd.to_numeric(df[c], errors='coerce')
        bad = ((s.notna()) & ((s < 1900) | (s > 2100))).sum()
        print(f"{c}_outside_[1900,2100]={int(bad)}")

# start/end chronology check (year-month-day to datetime)
def build_dt(prefix):
    y = pd.to_numeric(df[f'{prefix}Date_year'], errors='coerce') if f'{prefix}Date_year' in df.columns else pd.Series(np.nan, index=df.index)
    m = pd.to_numeric(df[f'{prefix}Date_month'], errors='coerce') if f'{prefix}Date_month' in df.columns else pd.Series(np.nan, index=df.index)
    d = pd.to_numeric(df[f'{prefix}Date_day'], errors='coerce') if f'{prefix}Date_day' in df.columns else pd.Series(np.nan, index=df.index)
    tmp = pd.DataFrame({'year': y, 'month': m.fillna(1), 'day': d.fillna(1)})
    return pd.to_datetime(tmp, errors='coerce')

if all(col in df.columns for col in ['startDate_year','startDate_month','startDate_day','endDate_year','endDate_month','endDate_day']):
    start_dt = build_dt('start')
    end_dt = build_dt('end')
    both = start_dt.notna() & end_dt.notna()
    bad_order = (both & (end_dt < start_dt)).sum()
    print(f'end_before_start={int(bad_order)} (among both_non_null={int(both.sum())})')

