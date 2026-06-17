"""
T2b：對 T1b 的 ablation delta 做 paired significance test
（回應老師 Q3「ablation delta 是否跨 seed 穩定/顯著，不是 seed noise」）。

對每個 (ablation, target, metric)：
  - 取 7 個配對 delta = ablated − full（同 seed，Run22–28 為 full）
  - one-sample t-test（H0: mean delta = 0）→ p_t（= paired t-test full vs ablated）
  - Wilcoxon signed-rank → p_w（非參數，n=7 較穩健）
  - 95% CI（t 分布）
  - p<0.05 且方向一致 → 標記「組件有顯著貢獻」

純統計，讀 runs/{run}/{target}/final_metrics.json，不重訓。
輸出：src_2/runs/t2b_significance.json + 印表。
用法：python src_2/t2b_significance.py
"""

import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

RUN = Path("src_2/runs")
SEED_FULL = {42: "22", 43: "23", 44: "24", 45: "25",
             247135: "26", 610172: "27", 796445: "28"}
ABLATIONS = ["ragoff", "noimg", "notrend"]
METRICS = {
    "popularity": ["log_MAE", "log_R2", "spearman_rho", "factor_acc_2x"],
    "meanScore":  ["MAE", "R2", "spearman_rho", "acc_within_10pt"],
}
LOWER_BETTER = {"log_MAE", "MAE"}   # 其餘越高越好


def _m(run_id, tg, k):
    p = RUN / run_id / tg / "final_metrics.json"
    return json.load(open(p)).get("test", {}).get(k) if p.exists() else None


def _deltas(abl, tg, k):
    """7 個配對 delta = ablated − full（同 seed）。"""
    out = []
    for seed, full in SEED_FULL.items():
        a = _m(f"t1b_{abl}_{seed}", tg, k)
        f = _m(full, tg, k)
        if a is not None and f is not None:
            out.append(float(a) - float(f))
    return np.array(out, dtype=float)


def main():
    results = {}
    print(f"{'ablation':9s} {'target':11s} {'metric':14s} | {'meanΔ':>9s} {'95% CI':>20s} | "
          f"{'t p':>8s} {'Wilcoxon p':>10s} | verdict")
    print("-" * 110)
    for abl in ABLATIONS:
        results[abl] = {}
        for tg, keys in METRICS.items():
            results[abl][tg] = {}
            for k in keys:
                d = _deltas(abl, tg, k)
                n = len(d)
                mean = float(d.mean())
                sem = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
                ci = (mean - tcrit * sem, mean + tcrit * sem)
                # paired t-test（= one-sample on deltas）
                t_stat, p_t = stats.ttest_1samp(d, 0.0)
                # Wilcoxon signed-rank（非參數）
                try:
                    _, p_w = stats.wilcoxon(d)
                except Exception:
                    p_w = float("nan")
                # 方向：error 指標 delta>0 = 組件有幫助；其餘 delta<0 = 組件有幫助
                helps = (mean > 0) if k in LOWER_BETTER else (mean < 0)
                sig = (p_t < 0.05)
                verdict = ("✓ 顯著貢獻" if (sig and helps) else
                           "✗ 顯著但反向" if (sig and not helps) else
                           "— 不顯著（seed noise）")
                results[abl][tg][k] = {
                    "n": n, "mean_delta": round(mean, 4),
                    "ci95": [round(ci[0], 4), round(ci[1], 4)],
                    "p_ttest": round(float(p_t), 4), "p_wilcoxon": round(float(p_w), 4),
                    "helps": bool(helps), "significant": bool(sig),
                }
                print(f"{abl:9s} {tg:11s} {k:14s} | {mean:+9.4f} "
                      f"[{ci[0]:+.3f},{ci[1]:+.3f}] | {p_t:8.4f} {p_w:10.4f} | {verdict}")
        print()

    out = RUN / "t2b_significance.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"✅ saved → {out}")

    # ── 重點摘要（primary metrics）──────────────────────────────────────────
    print("\n" + "=" * 70 + "\n重點摘要（primary：log_MAE/MAE 與 Spearman）\n" + "=" * 70)
    for abl in ABLATIONS:
        print(f"\n{abl}:")
        for tg, prim in [("popularity", ["log_MAE", "spearman_rho"]),
                         ("meanScore", ["MAE", "spearman_rho"])]:
            for k in prim:
                r = results[abl][tg][k]
                tag = "✓顯著" if r["significant"] and r["helps"] else \
                      "✗反向" if r["significant"] else "—不顯著"
                print(f"  {tg:11s} {k:13s}  Δ={r['mean_delta']:+.4f}  p={r['p_ttest']:.4f}  {tag}")


if __name__ == "__main__":
    main()
