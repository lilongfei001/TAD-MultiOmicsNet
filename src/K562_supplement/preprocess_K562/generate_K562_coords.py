import numpy as np
import pandas as pd

# Generate bin coordinate csv for K562
cell = "K562"
resolution = 10000    # Hi-C resolution (bp): 10000(10kb), 25000(25kb)
display_reso = int(resolution / 1000)
# Update this root_dir to your own project path before running
root_dir = "/mnt/d/TAD-MultiOmicsNet"

bin_size = resolution
chr_list = [f"chr{n}" for n in range(1, 23)]
hic_dir = f"{root_dir}/{cell}/{display_reso}kb/samples-generate/predict"
out_csv = f"{root_dir}/{cell}/{display_reso}kb/samples-generate/predict/{cell}_coords_{display_reso}kb.csv"

out_list = []
for ch in chr_list:
    hic_path = f"{hic_dir}/{cell}_{display_reso}kb_{ch}-matrix.npy"
    hic_mat = np.load(hic_path)
    n_bin = hic_mat.shape[0]

    coords = []
    for i in range(n_bin):
        s = i * bin_size
        e = s + bin_size
        coords.append([ch, s, e])

    df_ch = pd.DataFrame(coords, columns=["chr", "start", "end"])
    out_list.append(df_ch)
    print(f"{ch} generated bin count：{n_bin}")

df_total = pd.concat(out_list, axis=0, ignore_index=True)
df_total.to_csv(out_csv, index=False)
print(f"\nSaved coordinate file：{out_csv}")
print(f"Total bins：{len(df_total)}\n")
