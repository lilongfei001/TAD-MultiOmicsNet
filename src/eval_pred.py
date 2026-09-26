import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
)
# Load global project configurations
from config import root_dir, resolution, display_reso, SEED

cell = "GM12878"

# Module switches
use_epi = True            # Epigenetic features
use_dnabert = True        # DNABERT sequence features
use_curv = True           # DNA curvature features
use_transformer = False   # Transformer block
use_maskconv = True       # MaskConv module
use_gaussian = True       # Gaussian weighting module
use_adaptive_fuse = True  # Adaptive fusion module

# Auto-generate model name for distinguishing ablation experiments
name_parts = []
if use_transformer: name_parts.append("Tr")
if use_epi: name_parts.append("Ep")
if use_dnabert: name_parts.append("Dn")
if use_curv: name_parts.append("Cv")
if use_maskconv: name_parts.append("Mk")
if use_gaussian: name_parts.append("Gs")
if use_adaptive_fuse: name_parts.append("Af")
model_use = f"HiC_{'_'.join(name_parts)}_seed{SEED}"
print(f"\n{model_use}")


FIX_THRESH = 0.5
file_list = [
    ("chr20", f"{root_dir}/{cell}/{display_reso}kb/prediction_result/{model_use}/{display_reso}kb_{cell}_chr20-pred.npy"),
    ("chr21", f"{root_dir}/{cell}/{display_reso}kb/prediction_result/{model_use}/{display_reso}kb_{cell}_chr21-pred.npy"),
    ("chr22", f"{root_dir}/{cell}/{display_reso}kb/prediction_result/{model_use}/{display_reso}kb_{cell}_chr22-pred.npy"),
]

chr_prf1 = []
chr_auprc = []
chr_mcc = []
all_total_bins = 0

print("==== Per-chromosome evaluation ====")
for chr_name, path in file_list:
    arr = np.load(path)
    y_true = arr[:, -2]
    y_prob = arr[:, -1]
    y_pred = (y_prob >= FIX_THRESH)

    chr_total_bins = arr.shape[0]
    all_total_bins += chr_total_bins

    p = precision_score(y_true, y_pred, pos_label=1)
    r = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    mcc = matthews_corrcoef(y_true, y_pred)
    pos_num = np.sum(y_true == 1)
    pred_pos_num = np.sum(y_pred == 1)

    # AUPRC: skip if only one class exists
    if len(np.unique(y_true)) == 2:
        auprc = average_precision_score(y_true, y_prob)
    else:
        auprc = np.nan

    chr_prf1.append([p, r, f1])
    chr_auprc.append(auprc)
    chr_mcc.append(mcc)

    print(f"[{chr_name}] True boundaries:{pos_num};Pred boundaries:{pred_pos_num};P={p:.4f};R={r:.4f};F1={f1:.4f}")
    print(f"AUPRC={auprc:.4f};MCC={mcc:.4f}\n")

# Chromosome average metrics
chr_prf1_arr = np.array(chr_prf1)
chr_avg_p = np.nanmean(chr_prf1_arr[:, 0])
chr_avg_r = np.nanmean(chr_prf1_arr[:, 1])
chr_avg_f1 = np.nanmean(chr_prf1_arr[:, 2])
chr_avg_auprc = np.nanmean(chr_auprc)
chr_avg_mcc = np.nanmean(chr_mcc)

print("==== Chromosome average metrics (equal weight per chromosome) ====")
print(f"Average P={chr_avg_p:.4f};Average R={chr_avg_r:.4f};Average F1={chr_avg_f1:.4f}")
print(f"Average AUPRC={chr_avg_auprc:.4f};Average MCC={chr_avg_mcc:.4f}\n")

