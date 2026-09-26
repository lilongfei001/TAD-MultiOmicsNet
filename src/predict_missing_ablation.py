import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # Reduce TensorFlow log messages
os.environ["TF_DETERMINISTIC_OPS"] = "1"        # Enable deterministic operations for reproducibility
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"      # Fix cudnn randomness for reproducibility
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from model import init_model, fun
# Load global project configurations
from config import root_dir, resolution, display_reso, SEED

cell = "GM12878"
chrs = ["20", "21", "22"]
missing_rates_all = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Module switches
use_epi = True            # Epigenetic features
use_dnabert = True        # DNABERT sequence features
use_curv = True           # DNA curvature features
use_transformer = False   # Transformer block
use_maskconv = True       # MaskConv module
use_gaussian = False       # Gaussian weighting module
use_adaptive_fuse = False  # Adaptive fusion module

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
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

def apply_bin_missing(feat, orig_mask, idx_missing):
    feat_corrupted = feat.copy()
    mask_corrupted = orig_mask.copy()
    if len(idx_missing) > 0:
        if mask_corrupted.ndim == 2:
            mask_corrupted[idx_missing, :] = 0.0
        else:
            mask_corrupted[idx_missing] = 0.0
    return feat_corrupted, mask_corrupted


print(f"\n===== {model_use} =====")
model = init_model(
        use_epi=use_epi,
        use_dnabert=use_dnabert,
        use_curv=use_curv,
        use_transformer=use_transformer,
        use_maskconv=use_maskconv,
        use_gaussian=use_gaussian,
        use_adaptive_fuse=use_adaptive_fuse
        )
weight_path = f'{root_dir}/{cell}/{display_reso}kb/weights/{model_use}_best.h5'
model.load_weights(weight_path)
print(f"✅ Load weights: {weight_path}")

epi_feat_all = np.load(f"{root_dir}/{cell}/{display_reso}kb/samples-generate/predict/epi{cell}_predict_feat100.npy").astype("float32")
epi_mask_all = np.load(f"{root_dir}/{cell}/{display_reso}kb/samples-generate/predict/epi{cell}_predict_mask100.npy").astype("float32")
dna_feat_all = np.load(f"{root_dir}/{cell}/{display_reso}kb/samples-generate/predict/dna{cell}_predict_{display_reso}kb.npy").astype("float32")
dna_mask_all = np.load(f"{root_dir}/{cell}/{display_reso}kb/samples-generate/predict/dna{cell}_predict_mask_{display_reso}kb.npy").astype("float32")
curv_feat_all = np.load(f"{root_dir}/{cell}/{display_reso}kb/samples-generate/predict/curv{cell}_predict_feat100.npy").astype("float32")
curv_mask_all = np.load(f"{root_dir}/{cell}/{display_reso}kb/samples-generate/predict/curv{cell}_predict_mask100.npy").astype("float32")

mr_list = missing_rates_all
for mr in mr_list:
    print(f"\n------ missing_rate = {mr:.2f} ------")
    out_dir = Path(
        f'{root_dir}/{cell}/{display_reso}kb/prediction_result/missingRobust/{model_use}_mr{mr:.2f}')
    out_dir.mkdir(parents=True, exist_ok=True)
    ptr = 0
    for chrom_idx, chrom in enumerate(chrs):
        npy_path = f'{root_dir}/{cell}/{display_reso}kb/samples-generate/predict/{cell}_{display_reso}kb_chr{chrom}-matrix.npy'
        pre_data = np.load(npy_path)
        sample_num = pre_data.shape[0]
        data_hic_raw = pre_data[:, :-1].reshape(-1, 10, 10, 1).astype("float32")
        data_hic = fun(data_hic_raw)

        epi_feat_chr = epi_feat_all[ptr:ptr + sample_num].copy()
        epi_mask_chr = epi_mask_all[ptr:ptr + sample_num].copy()
        dna_feat_chr = dna_feat_all[ptr:ptr + sample_num].copy()
        dna_mask_chr = dna_mask_all[ptr:ptr + sample_num].copy()
        curv_feat_chr = curv_feat_all[ptr:ptr + sample_num].copy()
        curv_mask_chr = curv_mask_all[ptr:ptr + sample_num].copy()

        # Each (mr,chrom) has an independent random seed to ensure reproducibility.
        sub_seed = SEED * 1000 + int(mr * 100) + chrom_idx
        rng = np.random.default_rng(seed=sub_seed)
        n_bin = epi_feat_chr.shape[0]
        n_missing = int(n_bin * mr)
        idx_missing = rng.choice(np.arange(n_bin), size=n_missing,
                                 replace=False) if n_missing > 0 else np.array([], dtype=int)

        epi_feat_cor, epi_mask_cor = apply_bin_missing(epi_feat_chr, epi_mask_chr, idx_missing)
        dna_feat_cor, dna_mask_cor = apply_bin_missing(dna_feat_chr, dna_mask_chr, idx_missing)
        curv_feat_cor, curv_mask_cor = apply_bin_missing(curv_feat_chr, curv_mask_chr, idx_missing)

        zero_epi = np.count_nonzero(epi_mask_cor == 0.0) / epi_mask_cor.shape[0]
        zero_dna = np.count_nonzero(dna_mask_cor == 0.0) / dna_mask_cor.shape[0]
        zero_curv = np.count_nonzero(curv_mask_cor == 0.0) / curv_mask_cor.shape[0]
        print(f"DEBUG mr={mr}, chr={chrom}, zero_mask_frac: epi={zero_epi:.3f}, dna={zero_dna:.3f}, curv={zero_curv:.3f}")

        inputs = [data_hic]
        inputs.extend([epi_feat_cor, epi_mask_cor])
        inputs.extend([dna_feat_cor, dna_mask_cor])
        inputs.extend([curv_feat_cor, curv_mask_cor])
        ptr += sample_num

        predictions = model.predict(inputs, batch_size=64, verbose=1)
        pred_prob = np.array(predictions.flatten())
        merged = np.concatenate([pre_data, pred_prob.reshape(-1, 1)], axis=1)
        save_path = str(out_dir / f"{display_reso}kb_{cell}_chr{chrom}-pred.npy")
        np.save(save_path, merged)
        print(f"{save_path}, shape={merged.shape}")

print("\nPrediction file for total missing rate has been generated!")







