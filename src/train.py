import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Uncomment and set GPU ID if needed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # Reduce TensorFlow log messages
os.environ["TF_DETERMINISTIC_OPS"] = "1"        # Enable deterministic operations for reproducibility
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"      # Fix cudnn randomness for reproducibility

import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from model import init_model, fun

import cfg
cell = "GM12878"
resolution = 10000    # Hi-C resolution (bp): 10000(10kb), 25000(25kb)
display_reso = int(resolution / 1000)
SEED = 42  # 5 sets of random seeds:42,43,44,45,46
root_dir = "/mnt/d/TAD-MultiOmicsNet"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

use_epi = cfg.model_cfg["use_epi"]
use_dnabert = cfg.model_cfg["use_dnabert"]
use_curv = cfg.model_cfg["use_curv"]
use_transformer = cfg.model_cfg["use_transformer"]
use_maskconv = cfg.model_cfg["use_maskconv"]
use_gaussian = cfg.model_cfg["use_gaussian"]
use_adaptive_fuse = cfg.model_cfg["use_adaptive_fuse"]
model_use = cfg.get_model_name(SEED)

# Training Entry
if __name__ == '__main__':
    # training set
    print("===== training set: Hi-C =====")
    train_data = np.load(
        f"{root_dir}/{cell}/{display_reso}kb/samples-generate/train/{cell}_{display_reso}kb_train-matrix_seed{SEED}.npy")
    X_train_single = train_data[:, :-1].reshape(-1, 10, 10, 1).astype("float32")
    X_train_hic = fun(X_train_single)
    y_train = train_data[:, -1]
    total_train = len(X_train_hic)
    train_x = [X_train_hic]

    if use_epi:
        print("===== training set: Epigenetic features =====")
        epi_train_feat = np.load(
            f"{root_dir}/{cell}/{display_reso}kb/samples-generate/train/epi{cell}_train_feat100_seed{SEED}.npy")
        epi_train_mask = np.load(
            f"{root_dir}/{cell}/{display_reso}kb/samples-generate/train/epi{cell}_train_mask100_seed{SEED}.npy")
        epi_train_feat = epi_train_feat[:total_train].astype("float32")
        epi_train_mask = epi_train_mask[:total_train].astype("float32")
        print(f'epi_train_feat shape: {epi_train_feat.shape}, mask shape:{epi_train_mask.shape}')
        train_x.extend([epi_train_feat, epi_train_mask])

    if use_dnabert:
        print("===== training set: DNABERT sequence features =====")
        dna_train_feat = np.load(
            f"{root_dir}/{cell}/{display_reso}kb/samples-generate/train/dna{cell}_train_{display_reso}kb_seed{SEED}.npy").astype(
            "float32")
        dna_train_mask = np.load(
            f"{root_dir}/{cell}/{display_reso}kb/samples-generate/train/dna{cell}_train_mask_{display_reso}kb_seed{SEED}.npy").astype(
            "float32")
        dna_train_feat = dna_train_feat[:total_train]
        dna_train_mask = dna_train_mask[:total_train]
        print(f"dna train feat shape: {dna_train_feat.shape}, mask shape:{dna_train_mask.shape}")
        train_x.extend([dna_train_feat, dna_train_mask])

    if use_curv:
        print("===== training set: DNA curvature features =====")
        curv_train_feat = np.load(
            f"{root_dir}/{cell}/{display_reso}kb/samples-generate/train/curv{cell}_train_feat100_seed{SEED}.npy").astype("float32")
        curv_train_mask = np.load(
            f"{root_dir}/{cell}/{display_reso}kb/samples-generate/train/curv{cell}_train_mask100_seed{SEED}.npy").astype("float32")
        curv_train_feat = curv_train_feat[:total_train]
        curv_train_mask = curv_train_mask[:total_train]
        print(f"curv train feat:{curv_train_feat.shape}, mask:{curv_train_mask.shape}")
        train_x.extend([curv_train_feat, curv_train_mask])


    # validation set
    print("\n===== validation set: Hi-C =====")
    val_chroms = [f"chr{i}" for i in range(13, 20)]
    X_val_hic_list = []
    y_val_list = []

    for chrom in val_chroms:
        print(f"Loading {chrom} ...")
        hic_path = f"{root_dir}/{cell}/{display_reso}kb/samples-generate/validation/{cell}_{display_reso}kb_{chrom}-matrix.npy"
        chrom_data = np.load(hic_path)
        X_single = chrom_data[:, :-1].reshape(-1, 10, 10, 1).astype("float32")
        X_hic_chrom = fun(X_single)
        y_chrom = chrom_data[:, -1]
        X_val_hic_list.append(X_hic_chrom)
        y_val_list.append(y_chrom)

    X_val_hic = np.concatenate(X_val_hic_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    val_x = [X_val_hic]
    print(f"X_val_hic shape: {X_val_hic.shape}")
    print(f"y_val shape: {y_val.shape}")

    if use_epi:
        print("===== validation set: Epigenetic features =====")
        epi_val_feat = np.load(
            f"{root_dir}/{cell}/{display_reso}kb/samples-generate/validation/epi{cell}_validation_feat100.npy").astype(
            "float32")
        epi_val_mask = np.load(
            f"{root_dir}/{cell}/{display_reso}kb/samples-generate/validation/epi{cell}_validation_mask100.npy").astype(
            "float32")
        val_x.extend([epi_val_feat, epi_val_mask])
        print(f"epi_val_feat shape: {epi_val_feat.shape}")

    if use_dnabert:
        print("===== validation set: DNABERT sequence features =====")
        dna_val_feat = np.load(
            f"{root_dir}/{cell}/{display_reso}kb/samples-generate/validation/dna{cell}_validation_{display_reso}kb.npy").astype(
            "float32")
        dna_val_mask = np.load(
            f"{root_dir}/{cell}/{display_reso}kb/samples-generate/validation/dna{cell}_validation_mask_{display_reso}kb.npy").astype(
            "float32")
        val_x.extend([dna_val_feat, dna_val_mask])
        print(f"dna_val_feat shape: {dna_val_feat.shape}, mask shape:{dna_val_mask.shape}")

    if use_curv:
        print("===== training set: DNA curvature features =====")
        curv_val_feat = np.load(
            f"{root_dir}/{cell}/{display_reso}kb/samples-generate/validation/curv{cell}_validation_feat100.npy").astype("float32")
        curv_val_mask = np.load(
            f"{root_dir}/{cell}/{display_reso}kb/samples-generate/validation/curv{cell}_validation_mask100.npy").astype("float32")
        val_x.extend([curv_val_feat, curv_val_mask])
        print(f"curv_val_feat shape: {curv_val_feat.shape}")


    # Initialize the model
    model = init_model(
        use_epi=use_epi,
        use_dnabert=use_dnabert,
        use_curv=use_curv,
        use_transformer=use_transformer,
        use_maskconv=use_maskconv,
        use_gaussian=use_gaussian,
        use_adaptive_fuse=use_adaptive_fuse
    )


    # callback function
    save_best_path = f"{root_dir}/{cell}/{display_reso}kb/weights/{model_use}_best.h5"
    save_best_cb = keras.callbacks.ModelCheckpoint(filepath=save_best_path, monitor="val_f1",
                                                   mode="max",save_best_only=True,
                                                   save_weights_only=True, verbose=1)

    early_stop = keras.callbacks.EarlyStopping(monitor="val_f1", patience=15,
                                               mode="max", restore_best_weights=True,
                                               verbose=1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_f1", factor=0.75,
                                                  patience=3, cooldown=1,
                                                  min_lr=5e-6, verbose=1,
                                                  mode="max")
    callbacks = [save_best_cb, early_stop, reduce_lr]

    hist = model.fit(
        train_x, y_train,
        validation_data=(val_x, y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        shuffle=True
    )
    # Save training history
    np.save(f"{root_dir}/{cell}/{display_reso}kb/hist/hist_{model_use}.npy", hist.history)
