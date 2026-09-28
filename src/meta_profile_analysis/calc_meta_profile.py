import os
import pandas as pd
import numpy as np

cell = "GM12878"
resolution = 10000    # Hi‑C resolution (bp): 10000(10kb), 25000(25kb)
display_reso = int(resolution / 1000)
# Update this root_dir to your own project path before running
root_dir = "/mnt/d/TAD-MultiOmicsNet"

PROJ = f"{root_dir}/{cell}/{display_reso}kb"
BED_DIR = f"{PROJ}/enrich_bed"
OUT_DIR = f"{BED_DIR}/enrich_result"
os.makedirs(OUT_DIR, exist_ok=True)

BW_PEAKS = {
    "RAD21":    f"{root_dir}/{cell}/epi_signal/bio/RAD21_GM12878.narrowPeak",
    "SMC3":     f"{root_dir}/{cell}/epi_signal/bio/SMC3_GM12878.narrowPeak",
    "H3k9me3":  f"{root_dir}/{cell}/epi_signal/bio/H3k9me3_GM12878.broadPeak",
}

NEAR_HALF = 10000
TAG_HALF  = 20000
NORM = 10000

# meta‑profile ±200kb/10kb bin
PROFILE_HALF = 200000
PROFILE_BIN  = 10000
bin_offsets = np.arange(-PROFILE_HALF, PROFILE_HALF + PROFILE_BIN, PROFILE_BIN)
n_bins = len(bin_offsets)

def is_overlap(ps, pe, ws, we):
    return not ((pe <= ws) or (ps >= we))

def count_overlap_peaks(peak_list, ws, we):
    return sum(1 for ps, pe in peak_list if is_overlap(ps, pe, ws, we))

def load_narrowpeak(path):
    peaks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split("\t")
            chrom = p[0]
            if not chrom.startswith("chr"):
                chrom = "chr" + chrom
            peaks.append((chrom, int(p[1]), int(p[2])))
    return peaks

peak_by_chr = {}
for marker, path in BW_PEAKS.items():
    plist = load_narrowpeak(path)
    d = {}
    for c, s, e in plist:
        d.setdefault(c, []).append((s, e))
    peak_by_chr[marker] = d
    print(f"{marker}: {len(plist)} peaks loaded")

# core function
def run_enrichment(boundary_bed_path):
    boundaries = []
    with open(boundary_bed_path) as f:
        for line in f:
            chrom, s, e = line.strip().split("\t")[:3]
            s, e = int(s), int(e)
            if not chrom.startswith("chr"):
                chrom = "chr" + chrom
            center = (s + e) // 2
            boundaries.append((chrom, center))
    n = len(boundaries)

    all_res = []
    prof_rows = []
    for marker in BW_PEAKS:
        peaks = peak_by_chr[marker]
        d_list = []
        tagged = 0
        profile_sum = np.zeros(n_bins)

        for chrom, center in boundaries:
            pk = peaks.get(chrom, [])
            cnt_near = count_overlap_peaks(pk, center - NEAR_HALF, center + NEAR_HALF)
            d_list.append(cnt_near / (2 * NEAR_HALF / NORM))
            cnt_tag = count_overlap_peaks(pk, center - TAG_HALF, center + TAG_HALF)
            if cnt_tag > 0:
                tagged += 1
            for i, off in enumerate(bin_offsets):
                bc = center + off
                cnt_bin = count_overlap_peaks(pk, bc - PROFILE_BIN//2, bc + PROFILE_BIN//2)
                profile_sum[i] += cnt_bin / (PROFILE_BIN / NORM)

        all_res.append({
            "marker": marker, "n_boundary": n,
            "average_peak": round(np.mean(d_list), 4),
            "tagged_ratio": round(tagged / n, 4),
        })
        prow = {"marker": marker}
        for i, off in enumerate(bin_offsets):
            prow[f"bin_{int(off/1000)}kb"] = profile_sum[i] / n
        prof_rows.append(prow)

    return pd.DataFrame(all_res), pd.DataFrame(prof_rows)

# =========================Own model: average of 5 seeds===================================
SEEDS = [42, 43, 44, 45, 46]
self_meta_list = []
self_summary_list = []
for SEED in SEEDS:
    bed_path = f"{BED_DIR}/{cell}_{display_reso}kb_Full{SEED}_boundary.bed"
    if not os.path.exists(bed_path):
        print(f"[WARN] 跳过 seed {SEED}")
        continue
    s_df, m_df = run_enrichment(bed_path)
    self_summary_list.append(s_df)
    self_meta_list.append(m_df)
    print(f"seed {SEED}: done")

df_self_meta_all = pd.concat(self_meta_list, ignore_index=True)
bin_cols = [c for c in df_self_meta_all.columns if c.startswith("bin_")]
# Take the average of the 5‑seed meta profile
df_self_meta = df_self_meta_all.groupby("marker")[bin_cols].mean().reset_index()
df_self_meta["method"] = "TAD-MultiOmicsNet"

# ============================External comparison algorithm=================================
method_bed = {
    "deDoc":    f"{BED_DIR}/{cell}_{display_reso}kb_deDoc_boundary.bed",
    "TopDom":   f"{BED_DIR}/{cell}_{display_reso}kb_TopDom_boundary.bed",
    "Arrowhead":    f"{BED_DIR}/{cell}_{display_reso}kb_Arrowhead_boundary.bed",
    "DomainCaller (DI)":     f"{BED_DIR}/{cell}_{display_reso}kb_DI_boundary.bed",
    "SpectralTAD":    f"{BED_DIR}/{cell}_{display_reso}kb_SpectralTAD_boundary.bed",
}

ext_meta_list = []
for name, bed_p in method_bed.items():
    if not os.path.exists(bed_p):
        print(f"[WARN] Cannot find {bed_p}, skip {name}")
        continue
    _, m_df = run_enrichment(bed_p)
    m_df["method"] = name
    ext_meta_list.append(m_df)
    print(f"{name}: done")

df_multi_meta = pd.concat([df_self_meta] + ext_meta_list, ignore_index=True)
df_multi_meta.to_csv(f"{OUT_DIR}/multi_method_meta.csv", index=False)

print("\nCalculation completed! The csv file has been saved, and you can run the plotting script now.")
