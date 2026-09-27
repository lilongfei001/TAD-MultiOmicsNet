import numpy as np
import pandas as pd

cell = "GM12878"
resolution = 10000    # Hi-C resolution (bp): 10000(10kb), 25000(25kb)
display_reso = int(resolution / 1000)
root_dir = "/mnt/d/TAD-MultiOmicsNet"
bin_size = resolution

# Generate bin coordinate csv for validation and test set
task_configs = [
    {
        "chr_list": [f"chr{n}" for n in range(13, 20)],
        "hic_dir": f"{root_dir}/{cell}/{display_reso}kb/samples-generate/validation",
        "out_csv": f"{root_dir}/{cell}/{display_reso}kb/samples-generate/validation/hic_val_coords13-19_{display_reso}kb.csv"
    },
    {
        "chr_list": [f"chr{n}" for n in range(20, 23)] + ["chrX"],
        "hic_dir": f"{root_dir}/{cell}/{display_reso}kb/samples-generate/predict",
        "out_csv": f"{root_dir}/{cell}/{display_reso}kb/samples-generate/predict/hic_predict_coords20-X_{display_reso}kb.csv"
    }
]

for task in task_configs:
    out_list = []
    for ch in task["chr_list"]:
        hic_path = f"{task['hic_dir']}/GM12878_{display_reso}kb_{ch}-matrix.npy"
        hic_mat = np.load(hic_path)
        n_bin = hic_mat.shape[0]

        coords = []
        for i in range(n_bin):
            s = i * bin_size
            e = s + bin_size
            coords.append([ch, s, e])

        df_ch = pd.DataFrame(coords, columns=["chr", "start", "end"])
        out_list.append(df_ch)
        print(f"{ch} generated bin count: {n_bin}")

    df_total = pd.concat(out_list, axis=0, ignore_index=True)
    df_total.to_csv(task["out_csv"], index=False)
    print(f"\nSaved coordinate file: {task['out_csv']}")
    print(f"Total bins: {len(df_total)}\n")