import os
import pandas as pd
import numpy as np

cell = "GM12878"
resolution = 10000    # Hi‑C resolution (bp): 10000(10kb), 25000(25kb)
display_reso = int(resolution//1000)
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
# Background bins: ±180 ~ ±200 kb flanking regions
BG_BIN_IDX = [0, 1, 2, n_bins-3, n_bins-2, n_bins-1]
CENTER_BIN_IDX = n_bins // 2  # Center bin index

# ========================= Utility Functions =========================
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

# ========================= Pre-load Peaks =========================
peak_by_chr = {}
for marker, path in BW_PEAKS.items():
    plist = load_narrowpeak(path)
    d = {}
    for c, s, e in plist:
        d.setdefault(c, []).append((s, e))
    peak_by_chr[marker] = d
    print(f"{marker}: {len(plist)} peaks loaded")

# ========================= Core Function =========================
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
    per_boundary_enrich = {}

    for marker in BW_PEAKS:
        peaks = peak_by_chr[marker]
        d_list = []
        tagged = 0
        profile_sum = np.zeros(n_bins)
        single_enrich_list = []

        for chrom, center in boundaries:
            pk = peaks.get(chrom, [])
            cnt_near = count_overlap_peaks(pk, center - NEAR_HALF, center + NEAR_HALF)
            d_list.append(cnt_near / (2 * NEAR_HALF / NORM))
            cnt_tag = count_overlap_peaks(pk, center - TAG_HALF, center + TAG_HALF)
            if cnt_tag > 0:
                tagged += 1

            center_signal = count_overlap_peaks(pk, center - PROFILE_BIN/2, center + PROFILE_BIN/2) / (PROFILE_BIN / NORM)
            bg_signals = []
            for off in bin_offsets[BG_BIN_IDX]:
                bc = center + off
                bg_sig = count_overlap_peaks(pk, bc - PROFILE_BIN/2, bc + PROFILE_BIN/2) / (PROFILE_BIN / NORM)
                bg_signals.append(bg_sig)
            bg_signal = np.mean(bg_signals)
            if bg_signal > 0:
                single_enrich_list.append(center_signal / bg_signal)
            else:
                single_enrich_list.append(np.nan)

            for i, off in enumerate(bin_offsets):
                bc = center + off
                cnt_bin = count_overlap_peaks(pk, bc - PROFILE_BIN/2, bc + PROFILE_BIN/2)
                profile_sum[i] += cnt_bin / (PROFILE_BIN / NORM)

        per_boundary_enrich[marker] = single_enrich_list

        all_res.append({
            "marker": marker, "n_boundary": n,
            "average_peak": round(np.mean(d_list), 4),
            "tagged_ratio": round(tagged / n, 4),
        })
        prow = {"marker": marker}
        for i, off in enumerate(bin_offsets):
            prow[f"bin_{int(off/1000)}kb"] = profile_sum[i] / n
        prof_rows.append(prow)

    return pd.DataFrame(all_res), pd.DataFrame(prof_rows), per_boundary_enrich

# =========================bootstrap function==================================
def calc_bootstrap_ci(enrich_list, n_bootstrap=1000, ci=0.95):
    enrich_arr = np.array(enrich_list)
    enrich_arr = enrich_arr[~np.isnan(enrich_arr)]
    n = len(enrich_arr)
    if n == 0:
        return np.nan, np.nan, np.nan
    mean_val = np.mean(enrich_arr)
    boot_means = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        boot_means.append(np.mean(enrich_arr[idx]))
    low = (1 - ci) / 2
    high = 1 - low
    ci_low, ci_high = np.percentile(boot_means, [low*100, high*100])
    return round(mean_val, 4), round(ci_low, 4), round(ci_high, 4)

# =========================Own model: average of 5 seeds===================================
SEEDS = [42, 43, 44, 45, 46]
self_meta_list = []
self_summary_list = []
self_enrich_all = []
for SEED in SEEDS:
    bed_path = f"{BED_DIR}/{cell}_{display_reso}kb_Full{SEED}_boundary.bed"
    if not os.path.exists(bed_path):
        print(f"[WARN] skip seed {SEED}")
        continue
    s_df, m_df, enrich_per = run_enrichment(bed_path)
    self_summary_list.append(s_df)
    self_meta_list.append(m_df)
    self_enrich_all.append(enrich_per)
    print(f"seed {SEED}: done")

df_self_meta_all = pd.concat(self_meta_list, ignore_index=True)
bin_cols = [c for c in df_self_meta_all.columns if c.startswith("bin_")]
# Take the average of the 5‑seed meta profile
df_self_meta = df_self_meta_all.groupby("marker")[bin_cols].mean().reset_index()
df_self_meta["method"] = "TAD-MultiOmicsNet"

# ---------- bootstrap CI for our model (merge all 5 seeds) ----------
enrich_quant_result = []
for marker in BW_PEAKS:
    all_enrich = []
    for seed_enrich in self_enrich_all:
        all_enrich.extend(seed_enrich[marker])
    mean_enrich, ci_low, ci_high = calc_bootstrap_ci(all_enrich)
    enrich_quant_result.append({
        "method": "TAD-MultiOmicsNet",
        "marker": marker,
        "enrichment_fold": mean_enrich,
        "ci_low": ci_low,
        "ci_high": ci_high
    })

# ============================External comparison algorithm=================================
method_bed = {
    "deDoc":    f"{BED_DIR}/{cell}_{display_reso}kb_deDoc_boundary.bed",
    "TopDom":   f"{BED_DIR}/{cell}_{display_reso}kb_TopDom_boundary.bed",
    "Arrowhead":    f"{BED_DIR}/{cell}_{display_reso}kb_Arrowhead_boundary.bed",
    "DomainCaller (DI)":     f"{BED_DIR}/{cell}_{display_reso}kb_DI_boundary.bed",
    "SpectralTAD":    f"{BED_DIR}/{cell}_{display_reso}kb_SpectralTAD_boundary.bed",
}
ext_meta_list = []
ext_enrich_list = []

for name, bed_p in method_bed.items():
    if not os.path.exists(bed_p):
        print(f"[WARN] Cannot find {bed_p}, skip {name}")
        continue
    _, m_df, enrich_per = run_enrichment(bed_p)
    m_df["method"] = name
    ext_meta_list.append(m_df)

    # ---------- bootstrap CI for external methods ----------
    for marker in BW_PEAKS:
        mean_enrich, ci_low, ci_high = calc_bootstrap_ci(enrich_per[marker])
        enrich_quant_result.append({
            "method": name,
            "marker": marker,
            "enrichment_fold": mean_enrich,
            "ci_low": ci_low,
            "ci_high": ci_high
        })
    print(f"{name}: done")

# ========================= Save Results =========================
df_multi_meta = pd.concat([df_self_meta] + ext_meta_list, ignore_index=True)
df_multi_meta.to_csv(f"{OUT_DIR}/multi_method_meta.csv", index=False)

df_enrich_quant = pd.DataFrame(enrich_quant_result)
df_enrich_quant.to_csv(f"{OUT_DIR}/enrichment_fold_bootstrap.csv", index=False)

print("\nCalculation completed! Output files:")
print(f"- Meta-profile matrix：{OUT_DIR}/multi_method_meta.csv")
print(f"- Enrichment fold + 95% bootstrap CI：{OUT_DIR}/enrichment_fold_bootstrap.csv")
