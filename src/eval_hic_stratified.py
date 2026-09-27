import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
)

cell = "GM12878"
resolution = 10000    # Hi-C resolution (bp): 10000(10kb), 25000(25kb)
display_reso = int(resolution // 1000)
root_dir = "/mnt/d/TAD-MultiOmicsNet"

FIX_THRESH = 0.5
SEEDS = [42, 43, 44, 45, 46]
chr_list = ["chr20", "chr21", "chr22"]

ablation_settings = [
    {"use_epi": False, "use_dnabert": False, "use_curv": False},
    {"use_epi": True, "use_dnabert": False, "use_curv": False},
]
use_transformer = False
use_maskconv = True
use_gaussian = True
use_adaptive_fuse = True

def get_name(abl_set):
    name_parts = []
    if use_transformer: name_parts.append("Tr")
    if abl_set["use_epi"]:  name_parts.append("Ep")
    if abl_set["use_dnabert"]:  name_parts.append("Dn")
    if abl_set["use_curv"]: name_parts.append("Cv")
    if use_maskconv: name_parts.append("Mk")
    if use_gaussian: name_parts.append("Gs")
    if use_adaptive_fuse: name_parts.append("Af")
    return f"HiC_{'_'.join(name_parts)}"

model_name_list = [get_name(abl_set) for abl_set in ablation_settings]

def load_pred(model_name, SEED, chr_name):
    model_use = f"{model_name}_seed{SEED}"
    path = (f"{root_dir}/{cell}/{display_reso}kb/prediction_result/"
            f"{model_use}/{display_reso}kb_{cell}_{chr_name}-pred.npy")
    arr = np.load(path)
    y_true = arr[:, -2]
    y_prob = arr[:, -1]
    hic_mean = np.mean(arr[:, :100], axis=1)
    return hic_mean, y_true, y_prob

def eval_metrics(y_true, y_prob):
    y_pred = (y_prob >= FIX_THRESH).astype(int)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan
    f1 = f1_score(y_true, y_pred, pos_label=1)
    auprc = average_precision_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    return f1, auprc, mcc


rows = []
# Step 1: Stratify by chromosome and calculate metrics for each group of every chromosome
for SEED in SEEDS:
    for chr_name in chr_list:
        hic_base, yt_base, _ = load_pred(model_name_list[0], SEED, chr_name)

        q33 = np.quantile(hic_base, 0.33)
        q66 = np.quantile(hic_base, 0.66)
        groups = np.zeros(len(hic_base), dtype=int)
        groups[(hic_base > q33) & (hic_base <= q66)] = 1
        groups[hic_base > q66] = 2

        for exp_idx, m_name in enumerate(model_name_list):
            _, yt_exp, yp_exp = load_pred(m_name, SEED, chr_name)
            assert np.allclose(yt_exp, yt_base), f"exp[{exp_idx}] {chr_name} seed{SEED} label mismatch!"

            for gid, gname in [(0, "low_interaction"), (1, "medium_interaction"), (2, "high_interaction")]:
                mask = groups == gid
                f1, ap, mcc = eval_metrics(yt_exp[mask], yp_exp[mask])
                rows.append([
                    exp_idx, m_name, chr_name, gname, SEED, int(mask.sum()),
                    f1, ap, mcc
                ])
df = pd.DataFrame(rows, columns=[
    "exp_idx", "model_name", "chr", "group", "seed", "n_samples",
    "F1", "AUPRC", "MCC"
])

# For each SEED, first compute the average for the three chromosomes within the same group.
df_seed_avg = df.groupby(["exp_idx", "model_name", "group", "seed"], as_index=False).agg(
    n_samples=("n_samples", "mean"),
    F1=("F1", "mean"),
    AUPRC=("AUPRC", "mean"),
    MCC=("MCC", "mean")
)

# Calculate the mean ± standard deviation for the 5 SEEDs to obtain the final summary.
summary_rows = []
for exp_idx in range(len(model_name_list)):
    for gname in ["low_interaction", "medium_interaction", "high_interaction"]:
        sub = df_seed_avg[(df_seed_avg["exp_idx"] == exp_idx) & (df_seed_avg["group"] == gname)]
        summary_rows.append({
            "exp_idx": exp_idx,
            "model_name": model_name_list[exp_idx],
            "group": gname,
            "n_mean": sub["n_samples"].mean(),
            "F1": f"{sub['F1'].mean():.4f}±{sub['F1'].std():.4f}",
            "AUPRC": f"{sub['AUPRC'].mean():.4f}±{sub['AUPRC'].std():.4f}",
            "MCC": f"{sub['MCC'].mean():.4f}±{sub['MCC'].std():.4f}",
        })

df_sum = pd.DataFrame(summary_rows)
# Print result
print(f"{cell} {display_reso}kb hierarchical results of Hi-C test set")
print(df_sum.to_string(index=False))
