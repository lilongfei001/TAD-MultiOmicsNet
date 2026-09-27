import pandas as pd
import numpy as np
import os

cell = "GM12878"
resolution = 10000
display_reso = int(resolution / 1000)
root_dir = "/mnt/d/TAD-MultiOmicsNet"

alpha_dir = f"{root_dir}/{cell}/alpha"
boundary_dir = f"{root_dir}/{cell}/alpha"

modality_cols = ["HiC_alpha", "Epi_alpha", "DNABERT_alpha", "Curvature_alpha"]
seeds = [42, 43, 44, 45, 46]
def load_boundary(file_path, res):
    boundary_set = set()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                chrom = parts[0]
                bp_start = int(parts[1])
                bin_start = bp_start // res
                boundary_set.add((chrom, bin_start))
    return boundary_set

# Load strong and weak boundaries
strong_set = load_boundary(f"{boundary_dir}/{cell}_{display_reso}kb_strong_boundary.txt", resolution)
weak_set = load_boundary(f"{boundary_dir}/{cell}_{display_reso}kb_weak_boundary.txt", resolution)

seed_strong = []
seed_weak = []

for seed in seeds:
    file_path = f"{alpha_dir}/alpha_{display_reso}kb_HiC_Ep_Dn_Cv_Mk_Gs_Af_seed{seed}.csv"
    if not os.path.exists(file_path):
        print(f"Skip seed{seed}: File does not exist")
        continue

    df = pd.read_csv(file_path)
    df["bin_start"] = df["bin_start"].astype(int)
    df["key"] = list(zip(df["chrom"], df["bin_start"]))

    strong_df = df[df["key"].isin(strong_set)]
    weak_df = df[df["key"].isin(weak_set)]

    strong_mean = strong_df[modality_cols].mean().values
    weak_mean = weak_df[modality_cols].mean().values

    seed_strong.append(strong_mean)
    seed_weak.append(weak_mean)

    print(f"seed{seed} | Number of strong boundaries: {len(strong_df)} | Number of weak boundaries: {len(weak_df)}")

strong_mean_all = np.mean(seed_strong, axis=0)
strong_std_all = np.std(seed_strong, axis=0, ddof=1)
weak_mean_all = np.mean(seed_weak, axis=0)
weak_std_all = np.std(seed_weak, axis=0, ddof=1)

# Output result
print(f"\n{cell} {display_reso}kb Modal attention weights for different boundary types (mean±SD, n=5 random seeds)")
print(f"{'Boundary Type':<8}", end="\t")
for col in modality_cols:
    print(f"{col:<18}", end="\t")
print()

print(f"{'strong boundary':<8}", end="\t")
for m, s in zip(strong_mean_all, strong_std_all):
    print(f"{m:.4f} ± {s:.4f}  ", end="\t")
print()

print(f"{'weak boundary':<8}", end="\t")
for m, s in zip(weak_mean_all, weak_std_all):
    print(f"{m:.4f} ± {s:.4f}  ", end="\t")
print()

"""
print("\nMean value of each seed strong boundary mode：")
for idx, s_mean in enumerate(seed_strong):
    print(f"seed{seeds[idx]}: {s_mean.round(4)}")
print("Mean value of each seed weak boundary mode：")
for idx, w_mean in enumerate(seed_weak):
    print(f"seed{seeds[idx]}: {w_mean.round(4)}")
"""
