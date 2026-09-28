from dnacurve import CurvedDNA
import numpy as np
from pyfaidx import Fasta
import csv
import os

# Update this root_dir to your own project path before running
root_dir = "/mnt/d/TAD-MultiOmicsNet"

def calc_bin_curvature(bin_seq: str, max_len=500):
    seq_upper = bin_seq.upper()
    n_count = seq_upper.count("N")
    if n_count / len(seq_upper) > 0.90:
        return np.nan

    all_vals = []
    for i in range(0, len(bin_seq), max_len):
        sub_seq = bin_seq[i:i+max_len]
        sub_upper = sub_seq.upper()
        if sub_upper.count("N") / len(sub_upper) > 0.90:
            continue
        cdna = CurvedDNA(sub_seq)
        arr = cdna.curvature.flatten()
        valid = arr[arr != 0]
        if len(valid) > 0:
            all_vals.extend(valid.tolist())
    if len(all_vals) == 0:
        return np.nan
    return np.mean(all_vals)

FASTA_PATH = f"{root_dir}/hg19.fa"
BED_PATH = f"{root_dir}/all_1kb_bins.bed"
OUT_CSV = f"{root_dir}/DNAcurve/bin_dna_curvature.csv"
REPORT_STEP = 5000

fa = Fasta(FASTA_PATH)
skip = 0

resume_chr = None
resume_start = -1
pass_resume = True

if os.path.exists(OUT_CSV):
    last_row = None
    with open(OUT_CSV, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                last_row = row
    if last_row is not None:
        resume_chr = last_row[0]
        resume_start = int(last_row[1])
        pass_resume = False
        print(f"断点续跑，上次结束位置：{resume_chr}:{resume_start}")
else:
    with open(OUT_CSV, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["chr","start","end","curv_mean"])
    pass_resume = True

count = 0

with open(BED_PATH, "r") as f_in, open(OUT_CSV, "a", newline="") as f_out:
    writer = csv.writer(f_out)
    for line in f_in:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            skip += 1
            continue
        chr_, s_str, e_str = parts[0], parts[1], parts[2]
        try:
            s = int(s_str)
            e = int(e_str)
        except Exception:
            skip += 1
            continue

        # 断点跳过逻辑：到达上次坐标之后才开始计算写入
        if not pass_resume:
            if chr_ == resume_chr:
                if s <= resume_start:
                    continue
                else:
                    pass_resume = True
            else:
                continue

        try:
            seq = fa[chr_][s:e].seq
            curv = calc_bin_curvature(seq)
            writer.writerow([chr_, s, e, curv])
            count += 1
            if count % REPORT_STEP == 0:
                print(f"Processed {count} bins | skip:{skip} | current:{chr_}:{s}")
        except Exception as err:
            skip += 1
            print(f"Skip {chr_}:{s}-{e}, error: {str(err)}")

print(f"\nDone. Total processed bins: {count}, skipped bins: {skip}")