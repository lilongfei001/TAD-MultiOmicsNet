import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # Reduce TensorFlow log messages

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from model_debug_alpha import init_model_alpha, fun

cell = "GM12878"
resolution = 10000    # Hi-C resolution (bp): 10000(10kb), 25000(25kb)
display_reso = int(resolution / 1000)
# Update this root_dir to your own project path before running
root_dir = "/mnt/d/TAD-MultiOmicsNet"
SEEDS = [42, 43, 44, 45, 46]
test_chroms = ["chr20", "chr21", "chr22", "chrX"]
batch_size = 64

# ablation config, keep consistent with training
abl_cfg = {
    "use_epi": True,
    "use_dnabert": True,
    "use_curv": True,
    "use_transformer": False,
    "use_maskconv": True,
    "use_gaussian": True,
    "use_adaptive_fuse": True
}

def single_seed_infer(seed):
    # build model full name
    name_parts = []
    if abl_cfg["use_transformer"]: name_parts.append("Tr")
    if abl_cfg["use_epi"]: name_parts.append("Ep")
    if abl_cfg["use_dnabert"]: name_parts.append("Dn")
    if abl_cfg["use_curv"]: name_parts.append("Cv")
    if abl_cfg["use_maskconv"]: name_parts.append("Mk")
    if abl_cfg["use_gaussian"]: name_parts.append("Gs")
    if abl_cfg["use_adaptive_fuse"]: name_parts.append("Af")
    base_name = f"HiC_{'_'.join(name_parts)}"
    model_full_name = f"{base_name}_seed{seed}"

    weight_path = f"{root_dir}/{cell}/{display_reso}kb/weights/{model_full_name}_best.h5"
    print(f"\n========== Inference: {model_full_name} ==========")
    if not os.path.exists(weight_path):
        print(f"WARNING weight file not found: {weight_path}, skip seed {seed}")
        return

    # load Hi‑C per‑chromosome matrix
    X_val_hic_list = []
    chrom_sample_count = {}
    y_true_chrom_list = []
    for chrom in test_chroms:
        p = f"{root_dir}/{cell}/{display_reso}kb/samples-generate/predict/{cell}_{display_reso}kb_{chrom}-matrix.npy"
        arr = np.load(p)
        yt = arr[:, -1]
        Xh = arr[:, :-1].reshape(-1, 10, 10, 1).astype(np.float32)
        Xh = fun(Xh)
        X_val_hic_list.append(Xh)
        y_true_chrom_list.append(yt)
        chrom_sample_count[chrom] = Xh.shape[0]
    X_val_hic = np.concatenate(X_val_hic_list, axis=0)
    y_true_all = np.concatenate(y_true_chrom_list)
    total_samples = X_val_hic.shape[0]
    val_inputs = [X_val_hic]

    # load multi‑omics feature
    if abl_cfg["use_epi"]:
        epi_feat = np.load(f"{root_dir}/{cell}/{display_reso}kb/samples-generate/predict/epi{cell}_predict_feat100.npy").astype(np.float32)
        epi_mask = np.load(f"{root_dir}/{cell}/{display_reso}kb/samples-generate/predict/epi{cell}_predict_mask100.npy").astype(np.float32)
        val_inputs.extend([epi_feat, epi_mask])
    if abl_cfg["use_dnabert"]:
        dna_feat = np.load(f"{root_dir}/{cell}/{display_reso}kb/samples-generate/predict/dna{cell}_predict_{display_reso}kb.npy").astype(np.float32)
        dna_mask = np.load(f"{root_dir}/{cell}/{display_reso}kb/samples-generate/predict/dna{cell}_predict_mask_{display_reso}kb.npy").astype(np.float32)
        val_inputs.extend([dna_feat, dna_mask])
    if abl_cfg["use_curv"]:
        curv_feat = np.load(f"{root_dir}/{cell}/{display_reso}kb/samples-generate/predict/curv{cell}_predict_feat100.npy").astype(np.float32)
        curv_mask = np.load(f"{root_dir}/{cell}/{display_reso}kb/samples-generate/predict/curv{cell}_predict_mask100.npy").astype(np.float32)
        val_inputs.extend([curv_feat, curv_mask])

    # init dual‑output model
    model = init_model_alpha(
        use_epi=abl_cfg["use_epi"],
        use_dnabert=abl_cfg["use_dnabert"],
        use_curv=abl_cfg["use_curv"],
        use_transformer=abl_cfg["use_transformer"],
        use_maskconv=abl_cfg["use_maskconv"],
        use_gaussian=abl_cfg["use_gaussian"],
        use_adaptive_fuse=abl_cfg["use_adaptive_fuse"]
    )
    model.load_weights(weight_path)
    y_pred, alpha = model.predict(val_inputs, batch_size=batch_size, verbose=1)

    # build chrom / bin_start label
    chrom_labels = []
    bin_in_chrom = []
    for chrom in test_chroms:
        cnt = chrom_sample_count[chrom]
        chrom_labels.extend([chrom] * cnt)
        bin_in_chrom.extend(list(range(cnt)))
    assert len(chrom_labels) == alpha.shape[0], "chrom label length mismatch with alpha"

    # ========== 1. Save per‑chromosome pred‑npy (for eval_hic_stratified.py) ==========
    out_pred_dir = f"{root_dir}/{cell}/{display_reso}kb/prediction_result/{model_full_name}"
    os.makedirs(out_pred_dir, exist_ok=True)
    pos = 0
    for chrom in test_chroms:
        cnt = chrom_sample_count[chrom]
        arr_out = np.zeros((cnt, 2), dtype=np.float32)
        arr_out[:, 0] = y_true_all[pos:pos+cnt]
        arr_out[:, 1] = y_pred[pos:pos+cnt, 0]
        # fill 100‑col dummy for hic_mean reading compatibility
        save_npy = np.concatenate([np.zeros((cnt,100)), arr_out], axis=1)
        npy_path = os.path.join(out_pred_dir, f"{display_reso}kb_{cell}_{chrom}-pred.npy")
        np.save(npy_path, save_npy)
        pos += cnt
    print(f"Saved per‑chromosome npy → {out_pred_dir}")

    # ========== 2. Save alpha csv ==========
    mod_names = ["HiC"]
    if abl_cfg["use_epi"]: mod_names.append("Epi")
    if abl_cfg["use_dnabert"]: mod_names.append("DNABERT")
    if abl_cfg["use_curv"]: mod_names.append("Curvature")
    save_dict = {
        "chrom": chrom_labels,
        "bin_start": bin_in_chrom,
        "y_pred": y_pred.squeeze().astype(np.float32)
    }
    for idx, name in enumerate(mod_names):
        save_dict[f"{name}_alpha"] = alpha[:, idx].astype(np.float32)
    df_out = pd.DataFrame(save_dict)
    alpha_dir = f"{root_dir}/{cell}/alpha"
    os.makedirs(alpha_dir, exist_ok=True)
    csv_path = os.path.join(alpha_dir, f"alpha_{display_reso}kb_{model_full_name}.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"Saved alpha csv → {csv_path}")

if __name__ == "__main__":
    for seed in SEEDS:
        single_seed_infer(seed)
    print("\n✅ All inference finished.")

