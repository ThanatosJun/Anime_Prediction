"""
重畫 Exp2 消融對比圖，用正確數據（seed=42，與論文 Table 7 一致）。
每條 bar：灰色 = 基準（reference）；彩色 = 該設定比基準多出的誤差；末端標數值。

產生兩張圖（兩種消融分開，不混 A/B）：
  Figure 6  組件消融（leave-one-out）：基準 = CARMA(all, trend ON)，移除 trend / image / RAG。
  Figure 7  單模態天花板（全 trend OFF）：基準 = CARMA w/o trend（trend OFF 的 full），對 metadata / text / image only。
            —— 單模態列本來就 trend OFF，故用 trend OFF 的 full 當基準才公平（同 trend 狀態）。

輸出：src_2/docs_new/figures/fig6_component_ablation.png、fig7_single_modality.png
用法：python src_2/make_ablation_figs.py
"""

from pathlib import Path
import matplotlib.pyplot as plt

OUT = Path("src_2/docs_new/figures")
OUT.mkdir(parents=True, exist_ok=True)

GRAY, BLUE, ORANGE = "#8C8C8C", "#4472C4", "#E8743B"


def panel(ax, labels, vals, base, color, xlabel, xlim):
    ypos = list(range(len(vals)))[::-1]            # 反轉 → 第一個 label 在最上
    for y, v in zip(ypos, vals):
        ax.barh(y, base, color=GRAY, height=0.62, zorder=3)
        if v > base + 1e-9:
            ax.barh(y, v - base, left=base, color=color, height=0.62, zorder=3)
            ax.text(base * 0.5, y, f"{base:.4f}", va="center", ha="center",
                    color="white", fontsize=8, zorder=4)
        ax.text(v + xlim * 0.012, y, f"{v:.4f}", va="center", ha="left", fontsize=9, zorder=4)
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, xlim); ax.set_xlabel(xlabel, fontsize=10)
    ax.grid(axis="x", color="0.85", lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)


def make_fig(labels, pop, score, pop_base, score_base, pop_xlim, score_xlim, fname):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.6))
    panel(ax1, labels, pop,   pop_base,   BLUE,   "Popularity log_MAE (Error)", pop_xlim)
    panel(ax2, labels, score, score_base, ORANGE, "MeanScore MAE (Error)",      score_xlim)
    ax1.set_title("Popularity", fontsize=12, fontweight="bold", pad=10)
    ax2.set_title("MeanScore", fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(OUT / f"{fname}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {OUT / fname}.png")


# ── Figure 6：組件消融（leave-one-out，基準 = CARMA all, trend ON）──────────────
make_fig(
    labels=["CARMA (all)", "CARMA w/o trend\n(No Time Trend)",
            "w/o image\n(No Visual Features)", "w/o RA all\n(No History Retrieval)"],
    pop  =[0.8823, 0.9036, 0.9307, 0.9473],
    score=[7.5911, 7.8877, 8.6352, 8.1246],
    pop_base=0.8823, score_base=7.5911,
    pop_xlim=1.05, score_xlim=9.6,
    fname="fig6_component_ablation",
)

# ── Figure 7：單模態天花板（全 trend OFF，基準 = CARMA w/o trend）──────────────
make_fig(
    labels=["CARMA w/o trend\n(full, trend off)", "w/ metadata only", "w/ text only", "w/ image only"],
    pop  =[0.9036, 0.9524, 1.2705, 1.3106],
    score=[7.8877, 7.9560, 10.3885, 8.7938],
    pop_base=0.9036, score_base=7.8877,
    pop_xlim=1.45, score_xlim=11.2,
    fname="fig7_single_modality",
)

# ── 合併圖：上排 (a) 組件消融、下排 (b) 單模態，清楚區分 ────────────────────────
A_LAB = ["CARMA (all)", "CARMA w/o trend\n(No Time Trend)",
         "w/o image\n(No Visual Features)", "w/o RA all\n(No History Retrieval)"]
A_POP, A_SCORE = [0.8823, 0.9036, 0.9307, 0.9473], [7.5911, 7.8877, 8.6352, 8.1246]
B_LAB = ["CARMA w/o trend\n(full, trend off)", "w/ metadata only", "w/ text only", "w/ image only"]
B_POP, B_SCORE = [0.9036, 0.9524, 1.2705, 1.3106], [7.8877, 7.9560, 10.3885, 8.7938]

fig, ax = plt.subplots(2, 2, figsize=(12, 8.2))
fig.subplots_adjust(left=0.20, right=0.97, top=0.88, bottom=0.07, hspace=0.95, wspace=0.52)
panel(ax[0, 0], A_LAB, A_POP,   0.8823, BLUE,   "Popularity log_MAE (Error)", 1.05)
panel(ax[0, 1], A_LAB, A_SCORE, 7.5911, ORANGE, "MeanScore MAE (Error)",      9.6)
panel(ax[1, 0], B_LAB, B_POP,   0.9036, BLUE,   "Popularity log_MAE (Error)", 1.45)
panel(ax[1, 1], B_LAB, B_SCORE, 7.8877, ORANGE, "MeanScore MAE (Error)",      11.2)
fig.text(0.5, 0.925, "(a) Component ablation — leave-one-out (remove one component from CARMA)",
         ha="center", fontsize=13, fontweight="bold")
fig.text(0.5, 0.445, "(b) Single-modality ceiling — each modality alone (temporal trend off)",
         ha="center", fontsize=13, fontweight="bold")
fig.add_artist(plt.Line2D([0.05, 0.97], [0.475, 0.475], color="0.75", lw=1.0))  # 分隔線
fig.savefig(OUT / "fig_exp2_ablation_combined.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "fig_exp2_ablation_combined.pdf", bbox_inches="tight")
plt.close(fig)
print(f"saved → {OUT / 'fig_exp2_ablation_combined.png'}")
