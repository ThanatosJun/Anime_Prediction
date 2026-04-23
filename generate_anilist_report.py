import pandas as pd
import numpy as np
import ast
import re
from pathlib import Path

csv_path = Path(r"C:/Users/g1014308/Desktop/archive_4/anilist_anime_data_complete.csv")
report_path = Path(r"C:/Users/g1014308/Desktop/anilist_分析報告.md")
analysis_date = "2026-04-23"


def fmt_int(x):
    try:
        return f"{int(x):,}"
    except Exception:
        return "N/A"


def fmt_float(x, nd=4):
    try:
        if pd.isna(x):
            return "NaN"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "NaN"


def parse_genres_cell(v):
    if pd.isna(v):
        return []
    if isinstance(v, list):
        return [str(i).strip() for i in v if str(i).strip()]
    s = str(v).strip()
    if s == "" or s in {"[]", "[ ]", "nan", "None", "null"}:
        return []
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [str(i).strip() for i in obj if str(i).strip()]
    except Exception:
        pass
    s2 = s.strip("[]")
    parts = [p.strip().strip("\"'") for p in s2.split(",") if p.strip()]
    return [p for p in parts if p]


def add_md_table(lines, df):
    if df is None or df.empty:
        lines.append("（無資料）")
        lines.append("")
        return
    lines.append(df.to_markdown(index=False))
    lines.append("")


if not csv_path.exists():
    raise FileNotFoundError(f"找不到資料檔：{csv_path}")

df = pd.read_csv(csv_path)
rows, cols = df.shape
mem_bytes = int(df.memory_usage(deep=True).sum())
mem_mb = mem_bytes / (1024 ** 2)

lines = []
lines.append("# AniList Anime 資料分析報告")
lines.append("")
lines.append(f"- **資料檔路徑**：`{csv_path.as_posix()}`")
lines.append(f"- **分析日期**：{analysis_date}")
lines.append("")

lines.append("## 1) 資料概覽")
lines.append("")
lines.append(f"- 列數（rows）：**{rows:,}**")
lines.append(f"- 欄數（columns）：**{cols:,}**")
lines.append(f"- 記憶體估計（含字串深度）：**{mem_bytes:,} bytes（約 {mem_mb:.2f} MB）**")
lines.append("")

lines.append("## 2) 全欄位清單與 dtype")
lines.append("")
dtype_df = pd.DataFrame({
    "欄位": df.columns,
    "dtype": [str(df[c].dtype) for c in df.columns],
    "非缺失數": [int(df[c].notna().sum()) for c in df.columns],
    "缺失數": [int(df[c].isna().sum()) for c in df.columns],
})
add_md_table(lines, dtype_df)

lines.append("## 3) 缺失率前20欄")
lines.append("")
missing = pd.DataFrame({
    "欄位": df.columns,
    "缺失數": [int(df[c].isna().sum()) for c in df.columns],
})
missing["缺失率"] = missing["缺失數"] / len(df)
missing = missing.sort_values(["缺失率", "缺失數"], ascending=[False, False]).head(20)
missing["缺失率"] = (missing["缺失率"] * 100).map(lambda x: f"{x:.2f}%")
add_md_table(lines, missing)

lines.append("## 4) 目標相關欄位描述統計與分位數")
lines.append("")
target_cols = ["averageScore", "meanScore", "popularity", "favourites", "trending", "episodes", "duration"]
q_list = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
for col in target_cols:
    lines.append(f"### {col}")
    if col not in df.columns:
        lines.append("- 欄位不存在。")
        lines.append("")
        continue
    s = pd.to_numeric(df[col], errors="coerce")
    desc = {
        "count": s.count(),
        "mean": s.mean(),
        "std": s.std(),
        "min": s.min(),
        "max": s.max(),
    }
    qv = s.quantile(q_list)
    stat_df = pd.DataFrame({
        "指標": ["count", "mean", "std", "min", "1%", "5%", "25%", "50%", "75%", "95%", "99%", "max"],
        "數值": [
            desc["count"], desc["mean"], desc["std"], desc["min"],
            qv.get(0.01, np.nan), qv.get(0.05, np.nan), qv.get(0.25, np.nan),
            qv.get(0.50, np.nan), qv.get(0.75, np.nan), qv.get(0.95, np.nan),
            qv.get(0.99, np.nan), desc["max"],
        ]
    })
    stat_df["數值"] = stat_df.apply(lambda r: fmt_int(r["數值"]) if r["指標"] == "count" else fmt_float(r["數值"], 4), axis=1)
    add_md_table(lines, stat_df)

lines.append("## 5) 類別欄位分布（前10）")
lines.append("")
cat_cols = ["type", "status", "source", "format", "season", "rating", "countryOfOrigin"]
for col in cat_cols:
    lines.append(f"### {col}")
    if col not in df.columns:
        lines.append("- 欄位不存在。")
        lines.append("")
        continue
    vc = df[col].astype("string").fillna("<NA>").value_counts(dropna=False).head(10)
    out = pd.DataFrame({"值": vc.index.astype(str), "筆數": vc.values})
    out["占比"] = (out["筆數"] / len(df) * 100).map(lambda x: f"{x:.2f}%")
    add_md_table(lines, out)

lines.append("## 6) genres 欄位解析")
lines.append("")
if "genres" not in df.columns:
    lines.append("- 欄位 `genres` 不存在。")
    lines.append("")
else:
    parsed = df["genres"].apply(parse_genres_cell)
    empty_ratio = (parsed.apply(len) == 0).mean() * 100
    lines.append(f"- 空陣列（或無法解析為有效類別）比例：**{empty_ratio:.2f}%**")
    all_genres = parsed.explode().dropna()
    all_genres = all_genres[all_genres.astype(str).str.strip() != ""]
    if len(all_genres) == 0:
        lines.append("- 無可用 genres 值。")
        lines.append("")
    else:
        gvc = all_genres.astype(str).value_counts().head(20)
        gdf = pd.DataFrame({"genre": gvc.index, "出現次數": gvc.values})
        add_md_table(lines, gdf)

lines.append("## 7) 時間欄位與資料品質")
lines.append("")
start_cols = sorted([c for c in df.columns if re.match(r"^startDate_", c)])
end_cols = sorted([c for c in df.columns if re.match(r"^endDate_", c)])
lines.append(f"- startDate_* 欄位：{', '.join(start_cols) if start_cols else '不存在'}")
lines.append(f"- endDate_* 欄位：{', '.join(end_cols) if end_cols else '不存在'}")

for ycol in ["startDate_year", "endDate_year", "seasonYear"]:
    if ycol in df.columns:
        s = pd.to_numeric(df[ycol], errors="coerce")
        if s.notna().any():
            lines.append(f"- {ycol} 範圍：{int(s.min())} ~ {int(s.max())}（非缺失）")
        else:
            lines.append(f"- {ycol}：皆為缺失或不可解析。")
    else:
        lines.append(f"- {ycol} 欄位不存在。")

invalid_month_info = []
for mcol in ["startDate_month", "endDate_month"]:
    if mcol in df.columns:
        m = pd.to_numeric(df[mcol], errors="coerce")
        invalid = ((m.notna()) & (~m.between(1, 12))).sum()
        invalid_month_info.append(f"{mcol}: {int(invalid)}")
    else:
        invalid_month_info.append(f"{mcol}: 欄位不存在")
lines.append("- 非法月份筆數（不在 1~12）：" + "； ".join(invalid_month_info))

invalid_day_info = []
for dcol in ["startDate_day", "endDate_day"]:
    if dcol in df.columns:
        d = pd.to_numeric(df[dcol], errors="coerce")
        invalid = ((d.notna()) & (~d.between(1, 31))).sum()
        invalid_day_info.append(f"{dcol}: {int(invalid)}")
    else:
        invalid_day_info.append(f"{dcol}: 欄位不存在")
lines.append("- 非法日期筆數（不在 1~31）：" + "； ".join(invalid_day_info))

needed = ["startDate_year", "startDate_month", "startDate_day", "endDate_year", "endDate_month", "endDate_day"]
if all(c in df.columns for c in needed):
    sdt = pd.to_datetime(
        pd.DataFrame({
            "year": pd.to_numeric(df["startDate_year"], errors="coerce"),
            "month": pd.to_numeric(df["startDate_month"], errors="coerce"),
            "day": pd.to_numeric(df["startDate_day"], errors="coerce"),
        }),
        errors="coerce"
    )
    edt = pd.to_datetime(
        pd.DataFrame({
            "year": pd.to_numeric(df["endDate_year"], errors="coerce"),
            "month": pd.to_numeric(df["endDate_month"], errors="coerce"),
            "day": pd.to_numeric(df["endDate_day"], errors="coerce"),
        }),
        errors="coerce"
    )
    end_before_start = ((sdt.notna()) & (edt.notna()) & (edt < sdt)).sum()
    lines.append(f"- `end_before_start` 筆數：**{int(end_before_start)}**")
else:
    miss = [c for c in needed if c not in df.columns]
    lines.append(f"- 無法計算 `end_before_start`，缺少欄位：{', '.join(miss)}")
lines.append("")

lines.append("## 8) 唯一性與重複")
lines.append("")
dup_rows = int(df.duplicated().sum())
lines.append(f"- 重複列數（全欄位完全相同）：**{dup_rows:,}**")
for id_col in ["id", "idMal"]:
    if id_col in df.columns:
        s = df[id_col]
        non_null = int(s.notna().sum())
        nunique = int(s.nunique(dropna=True))
        dup_non_null = non_null - nunique
        lines.append(f"- `{id_col}`：非缺失 {non_null:,}、唯一值 {nunique:,}、非缺失重複值數 {dup_non_null:,}")
    else:
        lines.append(f"- `{id_col}` 欄位不存在。")
lines.append("")

lines.append("## 9) 給「上映後人氣與品質預測」的可執行資料清理與特徵工程清單")
lines.append("")
cleaning_steps = [
    "統一缺失值表示（空字串、Unknown、[]、0 的語意需分流），並保留缺失指示欄位（missing indicators）。",
    "對 averageScore、meanScore、popularity、favourites、trending 做 log1p 候選轉換以降低長尾影響。",
    "以 IQR 或分位數截尾（winsorization）處理極端值，避免少數爆量作品主導模型。",
    "時間特徵工程：由 start/end 日期建構上映年份、季度、是否完結、上映至今月數。",
    "類別欄位（type/status/source/format/season/rating/countryOfOrigin）採 One-Hot 或 Target Encoding（需交叉驗證防洩漏）。",
    "genres 解析後做 Multi-hot 編碼；可新增 genre 數量、主 genre、稀有 genre 指標。",
    "長度特徵：episodes、duration 建立總時長（episodes*duration）與分箱特徵。",
    "建立互動特徵：如 format×season、source×rating、country×genre。",
    "切分策略採時間切分（time-based split）以模擬真實預測場景，避免未來資訊外洩。",
    "評估指標建議同時使用 MAE/RMSE（迴歸）與 Spearman（排序相關）衡量熱門度/品質預測效果。"
]
for step in cleaning_steps:
    lines.append(f"- {step}")
lines.append("")

lines.append("## 10) 結論摘要")
lines.append("")
conclusions = []
conclusions.append(f"資料集規模為 {rows:,} 列、{cols:,} 欄，記憶體約 {mem_mb:.2f} MB。")
if not missing.empty:
    top_missing_col = missing.iloc[0]["欄位"]
    top_missing_rate = missing.iloc[0]["缺失率"]
    conclusions.append(f"缺失最嚴重欄位為 `{top_missing_col}`，缺失率 {top_missing_rate}。")
if "genres" in df.columns:
    conclusions.append("genres 欄位可解析為多標籤特徵，適合做 multi-hot 與主題聚合特徵。")
if "id" in df.columns:
    id_dup = int(df["id"].notna().sum() - df["id"].nunique(dropna=True))
    conclusions.append(f"id 非缺失重複值數為 {id_dup:,}，可用於主鍵一致性檢核。")
if len(conclusions) < 3:
    conclusions.append("建議優先完成缺失處理、時間欄位校正與類別編碼，再進行建模。")
conclusions = conclusions[:6]
for i, c in enumerate(conclusions, 1):
    lines.append(f"{i}. {c}")
lines.append("")

report_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Report generated: {report_path}")
print(f"Rows={rows}, Cols={cols}, MemoryMB={mem_mb:.2f}")
