import os
import pandas as pd

cell = "GM12878"
resolution = 10000    # Hi‑C resolution (bp): 10000(10kb), 25000(25kb)
display_reso = int(resolution / 1000)
# Update this root_dir to your own project path before running
root_dir = "/mnt/d/TAD-MultiOmicsNet"

# Five traditional TAD‑calling tools
caller_list = ["Arrowhead", "DI", "deDoc", "SpectralTAD", "TopDom"]

chrom_order = [f"chr{i}" for i in range(20, 23)]

for caller in caller_list:
    print(f"\n========== Processing {caller} ==========")
    input_dir = f"{root_dir}/{cell}/{display_reso}kb/all_TADs/{caller}/"
    boundary_out = f"{root_dir}/{cell}/{display_reso}kb/enrich_bed/{cell}_{display_reso}kb_{caller}_boundary.bed"
    os.makedirs(os.path.dirname(boundary_out), exist_ok=True)

    all_boundary = []
    if not os.path.exists(input_dir):
        print(f"Warning: {input_dir} not found, skip {caller}")
        continue

    for fname in os.listdir(input_dir):
        if fname.startswith(f"{cell}_{display_reso}kb_{caller}.chr"):
            chrom = fname.split(".")[-1]
            if chrom not in chrom_order:
                continue
            fpath = os.path.join(input_dir, fname)
            # read TAD result, skip header
            df = pd.read_csv(fpath, sep="\t", header=0, names=["start", "end"])

            # left boundary of TAD
            left = df[["start"]].copy()
            left["chrom"] = chrom
            left["pos_end"] = left["start"] + resolution
            left = left[["chrom", "start", "pos_end"]]
            left.columns = ["chrom", "pos", "pos_end"]

            # right boundary of TAD
            right = df[["end"]].copy()
            right["chrom"] = chrom
            right["pos_end"] = right["end"] + resolution
            right = right[["chrom", "end", "pos_end"]]
            right.columns = ["chrom", "pos", "pos_end"]

            chr_bound = pd.concat([left, right])
            chr_bound = chr_bound.drop_duplicates(subset=["chrom", "pos"], keep="first")
            all_boundary.append(chr_bound)
            print(f"{fname} read, unique boundaries: {len(chr_bound)}")

    if len(all_boundary) == 0:
        print(f"No valid data for {caller}, skip output")
        continue

    bound_df = pd.concat(all_boundary, ignore_index=True)
    bound_df["chrom"] = pd.Categorical(bound_df["chrom"], categories=chrom_order, ordered=True)
    bound_df = bound_df.sort_values(["chrom", "pos"]).reset_index(drop=True)

    bound_df.to_csv(boundary_out, sep="\t", index=False, header=False)
    print(f"{caller} boundary BED saved: {boundary_out}")
    print(f"Total boundary sites: {len(bound_df)}")

