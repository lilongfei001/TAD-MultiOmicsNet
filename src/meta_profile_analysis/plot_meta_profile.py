import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Update this root_dir to your own project path before running
root_dir = "/mnt/d/TAD-MultiOmicsNet"
cell = "GM12878"

PROFILE_HALF = 200000
PROFILE_BIN  = 10000
bin_offsets = np.arange(-PROFILE_HALF, PROFILE_HALF + PROFILE_BIN, PROFILE_BIN)
kb_x_full = bin_offsets / 1000
mask_200kb = (kb_x_full >= -200) & (kb_x_full <= 200)
kb_x_200 = kb_x_full[mask_200kb]

color_map = {
    "TAD-MultiOmicsNet": "#d62728",
    "deDoc":  "#1f77b4",
    "TopDom": "#ff7f0e",
    "Arrowhead":  "#2ca02c",
    "SpectralTAD":  "#8c564b",
    "DomainCaller (DI)":   "#9467bd",
}
plot_markers = ["RAD21", "SMC3", "H3k9me3"]
keep_methods = list(color_map.keys())
labels_abc = ["A", "B", "C", "D", "E", "F"]

# ---------------------- Load 10kb dataset ----------------------
resolution1 = 10000
display_reso1 = int(resolution1 / 1000)
PROJ1 = f"{root_dir}/{cell}/{display_reso1}kb"
BED_DIR1 = f"{PROJ1}/enrich_bed"
OUT_DIR1 = f"{BED_DIR1}/enrich_result"
df_10k = pd.read_csv(f"{OUT_DIR1}/multi_method_meta.csv")

# ---------------------- Load 25kb dataset ----------------------
resolution2 = 25000
display_reso2 = int(resolution2 / 1000)
PROJ2 = f"{root_dir}/{cell}/{display_reso2}kb"
BED_DIR2 = f"{PROJ2}/enrich_bed"
OUT_DIR2 = f"{BED_DIR2}/enrich_result"
df_25k = pd.read_csv(f"{OUT_DIR2}/multi_method_meta.csv")

df_list = [df_10k, df_25k]

# Y‑axis limit configuration
ylim_settings = [
    {"RAD21":(0.07, 0.62), "SMC3":(0.07, 0.62), "H3k9me3":(0.42, 0.82)}, #10kb
    {"RAD21":(0.07, 0.62), "SMC3":(0.07, 0.62), "H3k9me3":(0.42, 0.82)}  #25kb
]

# ========== 2‑row × 3‑column figure ==========
fig, axes = plt.subplots(
    nrows=2, ncols=3,
    figsize=(12, 7.5),
    sharex="row",
    squeeze=False
)

plot_idx = 0
for res_idx, df_meta in enumerate(df_list):
    for col_idx, mk in enumerate(plot_markers):
        ax = axes[res_idx][col_idx]
        for meth in keep_methods:
            row = df_meta[(df_meta["marker"] == mk) & (df_meta["method"] == meth)]
            if row.empty:
                continue
            curve = row[[c for c in df_meta.columns if c.startswith("bin_")]].values[0]
            curve_200 = curve[mask_200kb]
            c = color_map.get(meth, "#7f7f7f")
            lw = 2.2 if meth == "TAD‑MultiOmicsNet" else 1.2
            ax.plot(kb_x_200, curve_200, label=meth, color=c, lw=lw)

        ax.axvline(0, color="k", ls="--", lw=0.8)
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin

        # Panel label A‑F, placed outside upper‑left corner
        ax.text(
            -0.17, 1.02,
            labels_abc[plot_idx],
            fontsize=13, fontweight="bold",
            va="bottom", ha="left",
            transform=ax.transAxes,
            clip_on=False
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_ylim(*ylim_settings[res_idx][mk])
        plot_idx += 1

# ========== Column titles (top of each column) ==========
for col_idx, mk in enumerate(plot_markers):
    ax = axes[0][col_idx]
    ax.text(
        0.5, 1.06, mk,
        fontsize=12, fontweight="bold",
        va="bottom", ha="center",
        transform=ax.transAxes
    )

# ========== Row titles (left side of each row) ==========
for row_idx, res_text in enumerate(["10 kb resolution", "25 kb resolution"]):
    ax = axes[row_idx][0]
    ax.text(
        -0.18, 0.5, res_text,
        fontsize=11,
        va="center", ha="right",
        rotation=90,
        transform=ax.transAxes
    )

# ========== Shared legend at bottom center ==========
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.01),
    ncol=6,
    frameon=False,
    fontsize=9,
    handlelength=1.5,
    columnspacing=1.2
)

# ========== Axis labels ==========
for i in range(3):
    axes[1][i].set_xlabel("Distance to TAD boundary (kb)", fontsize=11)

fig.text(
    0.01, 0.5, "Peak density (per 10 kb)",
    va="center", rotation="vertical", fontsize=11
)

# ========== Layout adjustment: reserve space for outer titles ==========
plt.tight_layout(rect=[0.06, 0.06, 0.99, 0.94])
plt.subplots_adjust(wspace=0.28, hspace=0.32)

# ========== Save output figure ==========
out_png = f"fig_multi_caller_10k_25k_2x3.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
plt.close()
print("\n✅ 2‑by‑3 composite figure saved:", out_png)








