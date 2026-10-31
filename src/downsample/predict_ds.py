import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# 依赖导入
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from modelTrain_ds import init_model, fun

cell = "GM12878"
resolution = 25000  # 10000,25000
display_reso = int(resolution // 1000)
root_dir = "/mnt/d/TAD-MultiOmicsNet"

DS_TAG = ["10r1", "10r2", "10r3", "25r1", "25r2", "25r3", "50r1", "50r2", "50r3", "75r1", "75r2", "75r3"]
ablation_settings = [
    {"use_epi": False, "use_dnabert": False, "use_curv": False, "use_trans": False},
    {"use_epi": True, "use_dnabert": False, "use_curv": False, "use_trans": False},
]
use_maskconv=True
use_gaussian=True
use_adaptive_fuse=True


for ds_tag in DS_TAG:
    print(f"=========================={ds_tag}====================")
    for cfg in ablation_settings:
        for SEED in [42, 43, 44, 45, 46]:
            use_transformer = cfg["use_trans"]
            use_epi = cfg["use_epi"]
            use_dnabert = cfg["use_dnabert"]
            use_curv = cfg["use_curv"]

            name_parts = []
            if use_transformer: name_parts.append("Tr")
            if use_epi: name_parts.append("Ep")
            if use_dnabert: name_parts.append("Dn")
            if use_curv: name_parts.append("Cv")

            parts_str = "_".join(name_parts)
            if parts_str:
                if parts_str == "Tr":
                    model_use = f"HiC_Tr_seed{SEED}"
                else:
                    model_use = f"HiC_{parts_str}_Mk_Gs_Af_seed{SEED}"
            else:
                model_use = f"HiC_seed{SEED}"
            # ===========================================

            random.seed(SEED)
            np.random.seed(SEED)
            tf.random.set_seed(SEED)

            out_dir = Path(f'{root_dir}/{cell}/sample/{display_reso}kb/{ds_tag}/prediction_result/{model_use}')
            out_dir.mkdir(parents=True, exist_ok=True)

            model = init_model(
                use_epi=use_epi,
                use_dnabert=use_dnabert,
                use_curv=use_curv,
                use_transformer=use_transformer,
                use_maskconv=use_maskconv,
                use_gaussian=use_gaussian,
                use_adaptive_fuse=use_adaptive_fuse
            )

            weight_path = f'{root_dir}/{cell}/sample/{display_reso}kb/{ds_tag}/weights/{model_use}_best.h5'
            model.load_weights(weight_path)
            print(f"Weights loaded successfully：{weight_path}")

            chrs = ["20", "21", "22", "X"]

            epi_feat_all, epi_mask_all = None, None
            dna_feat_all, dna_mask_all = None, None
            curv_feat_all, curv_mask_all = None, None

            ptr = 0
            if use_epi:
                epi_feat_all = np.load(
                    f"{root_dir}/{cell}/{display_reso}kb/samples-generate/predict/epi{cell}_predict_feat100.npy").astype(
                    "float32")
                epi_mask_all = np.load(
                    f"{root_dir}/{cell}/{display_reso}kb/samples-generate/predict/epi{cell}_predict_mask100.npy").astype(
                    "float32")
                print(f"Global epigenetic feature shape: {epi_feat_all.shape}, mask shape:{epi_mask_all.shape}")

            if use_dnabert:
                dna_feat_all = np.load(
                    f"{root_dir}/{cell}/{display_reso}kb/samples-generate/predict/dna{cell}_predict_{display_reso}kb.npy").astype(
                    "float32")
                dna_mask_all = np.load(
                    f"{root_dir}/{cell}/{display_reso}kb/samples-generate/predict/dna{cell}_predict_mask_{display_reso}kb.npy").astype(
                    "float32")
                print(f"Global DNABERT sequence feature shape: {dna_feat_all.shape}, mask shape:{dna_mask_all.shape}")

            if use_curv:
                curv_feat_all = np.load(
                    f"{root_dir}/{cell}/{display_reso}kb/samples-generate/predict/curv{cell}_predict_feat100.npy").astype(
                    "float32")
                curv_mask_all = np.load(
                    f"{root_dir}/{cell}/{display_reso}kb/samples-generate/predict/curv{cell}_predict_mask100.npy").astype(
                    "float32")
                print(f"Global DNA curvature feature shape: {curv_feat_all.shape}, mask shape:{curv_mask_all.shape}")

            for chrom in chrs:
                npy_path = f'{root_dir}/{cell}/sample/{display_reso}kb/{ds_tag}/samples-generate/predict/{cell}_{display_reso}kb_chr{chrom}_{ds_tag}.npy'
                pre_data = np.load(npy_path)
                sample_num = pre_data.shape[0]
                print(f"\n[{cell}] {display_reso}kb chr{chrom} raw shape: {pre_data.shape} 样本数={sample_num}")

                data_hic_raw = pre_data[:, :-1].reshape(-1, 10, 10, 1).astype("float32")
                data_hic = fun(data_hic_raw)

                inputs = [data_hic]
                if use_epi:
                    data_epi_feat = epi_feat_all[ptr:ptr + sample_num]
                    data_epi_mask = epi_mask_all[ptr:ptr + sample_num]
                    inputs.extend([data_epi_feat, data_epi_mask])
                if use_dnabert:
                    data_dna_feat = dna_feat_all[ptr:ptr + sample_num]
                    data_dna_mask = dna_mask_all[ptr:ptr + sample_num]
                    inputs.extend([data_dna_feat, data_dna_mask])
                if use_curv:
                    data_curv_feat = curv_feat_all[ptr:ptr + sample_num]
                    data_curv_mask = curv_mask_all[ptr:ptr + sample_num]
                    inputs.extend([data_curv_feat, data_curv_mask])

                ptr += sample_num

                batch_size = 64
                predictions = model.predict(inputs, batch_size=batch_size, verbose=1)
                predictions1 = np.array(predictions.flatten())

                assert len(predictions1) == sample_num, f"The number of predicted samples for chr{chrom} does not match!"

                merged_data = np.concatenate((pre_data, predictions1.reshape(-1, 1)), axis=1)
                save_path = str(out_dir / f'{display_reso}kb_{cell}_chr{chrom}_{ds_tag}.npy')
                np.save(save_path, merged_data)
                print(f"Saved → {save_path}, shape={merged_data.shape}")

            print("\nAll chromosome prediction tasks finished!")