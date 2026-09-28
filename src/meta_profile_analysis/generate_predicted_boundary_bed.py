import os
import numpy as np
import pandas as pd
import random

cell = "GM12878"
resolution = 10000    # Hi-C resolution (bp): 10000(10kb), 25000(25kb)
display_reso = int(resolution//1000)
# Update this root_dir to your own project path before running
root_dir = "/mnt/d/TAD-MultiOmicsNet"

chr_orders = [f"chr{i}" for i in range(20, 23)]
SEED_LIST = [42, 43, 44, 45, 46]

bed_out_dir = f"{root_dir}/{cell}/{display_reso}kb/enrich_bed/"
os.makedirs(bed_out_dir, exist_ok=True)

for SEED in SEED_LIST:
    model_use = f"HiC_Ep_Dn_Cv_Mk_Gs_Af_seed{SEED}"
    pred_dir = f"{root_dir}/{cell}/{display_reso}kb/prediction_result/{model_use}"
    threshold = 0.5

    out_bed = os.path.join(bed_out_dir, f"{cell}_{display_reso}kb_Full{SEED}_boundary.bed")

    all_boundaries = []
    # Collect all candidate bins with value >=0.5
    for chrom in chr_orders:
        fname = f"{display_reso}kb_{cell}_{chrom}-pred.npy"
        npy_path = os.path.join(pred_dir, fname)
        print(f"Processing {fname}")
        arr = np.load(npy_path)
        pred_prob = arr[:, -1]
        mask = pred_prob >= threshold
        bin_idx = np.where(mask)[0]
        prob_vals = pred_prob[mask]
        start = bin_idx * resolution
        end = start + resolution
        for s, e, p in zip(start, end, prob_vals):
            all_boundaries.append([chrom, s, e, p])

    df = pd.DataFrame(all_boundaries, columns=["chr", "start", "end", "prob"])
    df = df.sort_values(["chr", "start"]).reset_index(drop=True)
    # Continuous bin grouping, retaining the bin with the highest probability for each group
    df["group"] = (df["chr"] != df["chr"].shift()) | (df["start"] != df["end"].shift())
    df["group"] = df["group"].cumsum()
    df_filtered = df.loc[df.groupby("group")["prob"].idxmax()].sort_values(["chr", "start"])

    with open(out_bed, "w") as f:
        for _, row in df_filtered.iterrows():
            f.write(f"{row['chr']}\t{row['start']}\t{row['end']}\n")

    print(f"Model {model_use}, total candidate bins(>=0.5): {len(df)}, selected peak boundaries: {len(df_filtered)}")
