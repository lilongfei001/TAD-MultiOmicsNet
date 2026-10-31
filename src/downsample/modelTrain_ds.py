import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Uncomment and set GPU ID if needed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"


import math
import csv
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import multiply
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import L2

FOCAL_GAMMA = 2.0
LR_MULT = 3
INIT_LR = LR_MULT * 1e-4
GLOBAL_DROPOUT = 0.3
cell = "GM12878"
resolution = 25000  # 10000,25000
display_reso = int(resolution // 1000)
root_dir = "/mnt/d/TAD-MultiOmicsNet"

ablation_settings = [
    {"use_epi": True, "use_dnabert": False, "use_curv": False, "use_trans": False},
    {"use_epi": False, "use_dnabert": False, "use_curv": False, "use_trans": False},
]
DS_TAG = ["10r1", "10r2", "10r3", "25r1", "25r2", "25r3", "50r1", "50r2", "50r3", "75r1", "75r2", "75r3"]

use_maskconv=True
use_gaussian=True
use_adaptive_fuse=True

for ds_tag in DS_TAG:
    for cfg in ablation_settings:
        for SEED in [42, 43, 44, 45, 46]:
            batch_size = 64
            random.seed(SEED)
            np.random.seed(SEED)
            tf.random.set_seed(SEED)

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
            print(f"==============={model_use}+{ds_tag}====================")


            # ==============================================================================
            # CBAM module
            # Code sourced from DeepTAD open-source repository
            # ==============================================================================
            def cbam_block(cbam_feature, ratio=8, kernel_size=(3, 3)):
                cbam_feature = channel_attention(cbam_feature, ratio)
                cbam_feature = spatial_attention(cbam_feature, kernel_size)
                return cbam_feature

            def channel_attention(input_feature, ratio=8):
                channel = input_feature.shape[-1]
                filters = max(1, int(channel // ratio))
                shared_layer_one = tf.keras.layers.Dense(filters,
                                                         activation='relu',
                                                         kernel_initializer='he_normal',
                                                         use_bias=True,
                                                         bias_initializer='zeros')
                shared_layer_two = tf.keras.layers.Dense(channel,
                                                         kernel_initializer='he_normal',
                                                         use_bias=True,
                                                         bias_initializer='zeros')

                avg_pool = tf.keras.layers.GlobalAveragePooling2D()(input_feature)
                avg_pool = tf.keras.layers.Reshape((1, 1, channel))(avg_pool)
                avg_pool = shared_layer_one(avg_pool)
                avg_pool = shared_layer_two(avg_pool)

                max_pool = tf.keras.layers.GlobalMaxPooling2D()(input_feature)
                max_pool = tf.keras.layers.Reshape((1, 1, channel))(max_pool)
                max_pool = shared_layer_one(max_pool)
                max_pool = shared_layer_two(max_pool)

                cbam_feature = tf.keras.layers.Add()([avg_pool, max_pool])
                cbam_feature = tf.keras.layers.Activation('sigmoid')(cbam_feature)
                return multiply([input_feature, cbam_feature])

            def spatial_attention(input_feature, kernel_siz):
                kernel_size = kernel_siz
                channel = input_feature.shape[-1]
                cbam_feature = input_feature
                avg_pool = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
                max_pool = tf.keras.layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
                concat = tf.keras.layers.Concatenate(axis=3)([avg_pool, max_pool])
                cbam_feature = tf.keras.layers.Conv2D(filters=1,
                                                      kernel_size=kernel_size,
                                                      strides=1,
                                                      padding='same',
                                                      activation='sigmoid',
                                                      kernel_initializer='he_normal',
                                                      use_bias=False)(concat)
                return multiply([input_feature, cbam_feature])


            # ==============================================================================
            # Masked 1D Global Average Pooling Layer
            # ==============================================================================
            class MaskConv1D(tf.keras.layers.Layer):
                def __init__(self, filters, kernel_size, padding="same", activation=None, min_denom=1.5,**kwargs):
                    super().__init__(**kwargs)
                    self.filters = filters
                    self.kernel_size = kernel_size
                    self.padding = padding
                    self.activation = activation
                    self.min_denom = min_denom

                    self.act = tf.keras.layers.Activation(activation) if activation else None
                    self.conv = layers.Conv1D(filters, kernel_size, padding=padding, use_bias=True)
                    self.mask_counter = layers.Conv1D(1, kernel_size, padding=padding,
                                                      kernel_initializer="ones", bias_initializer="zeros", trainable=False)

                def call(self, inputs):
                    feat, mask = inputs
                    mask_exp = tf.expand_dims(mask, axis=-1)
                    conv_out = self.conv(feat)
                    count = self.mask_counter(mask_exp)

                    denominator = tf.maximum(count, self.min_denom)
                    out = conv_out / denominator

                    zero_mask = tf.cast(count > 1e-8, tf.float32)
                    out = out * zero_mask

                    if self.act is not None:
                        out = self.act(out)
                    return out

                def get_config(self):
                    cfg = super().get_config()
                    cfg.update({
                        "filters": self.filters,
                        "kernel_size": self.kernel_size,
                        "padding": self.padding,
                        "activation": self.activation,
                        "min_denom": self.min_denom
                    })
                    return cfg


            # ==============================================================================
            # F1-score
            # ==============================================================================
            class F1Score(tf.keras.metrics.Metric):
                def __init__(self, threshold=0.5, name="f1", **kwargs):
                    super().__init__(name=name, **kwargs)
                    self.threshold = threshold
                    self.tp = self.add_weight(name="tp", initializer="zeros")
                    self.fp = self.add_weight(name="fp", initializer="zeros")
                    self.fn = self.add_weight(name="fn", initializer="zeros")

                def update_state(self, y_true, y_pred, sample_weight=None):
                    y_pred_bin = tf.cast(tf.greater_equal(y_pred, self.threshold), tf.float32)
                    y_true = tf.cast(y_true, tf.float32)
                    tp_mask = y_true * y_pred_bin
                    fp_mask = (1. - y_true) * y_pred_bin
                    fn_mask = y_true * (1. - y_pred_bin)
                    if sample_weight is not None:
                        sample_weight = tf.cast(sample_weight, tf.float32)
                        tp_mask *= sample_weight
                        fp_mask *= sample_weight
                        fn_mask *= sample_weight
                    self.tp.assign_add(tf.reduce_sum(tp_mask))
                    self.fp.assign_add(tf.reduce_sum(fp_mask))
                    self.fn.assign_add(tf.reduce_sum(fn_mask))

                def result(self):
                    eps = tf.keras.backend.epsilon()
                    precision = self.tp / (self.tp + self.fp + eps)
                    recall = self.tp / (self.tp + self.fn + eps)
                    f1 = 2 * precision * recall / (precision + recall + eps)
                    return f1

                def get_pr(self):
                    eps = tf.keras.backend.epsilon()
                    precision = self.tp / (self.tp + self.fp + eps)
                    recall = self.tp / (self.tp + self.fn + eps)
                    return precision.numpy(), recall.numpy()

                def reset_state(self):
                    self.tp.assign(0.)
                    self.fp.assign(0.)
                    self.fn.assign(0.)


            # ==============================================================================
            # Hi-C CBAM-CNN branch
            # ==============================================================================
            def init_cnn():
                inputs = layers.Input(shape=(10, 10, 1))
                x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=L2(1e-4))(inputs)  # 添加L2正则
                x = layers.MaxPooling2D((2, 2))(x)
                x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=L2(1e-4))(x)
                x = layers.MaxPooling2D((2, 2))(x)
                x = cbam_block(x)
                x = layers.Dropout(GLOBAL_DROPOUT)(x)
                x = layers.Flatten()(x)
                x = layers.Dense(128, activation='relu', kernel_regularizer=L2(1e-4))(x)
                x = layers.LayerNormalization()(x)
                return tf.keras.Model(inputs, x)


            # ==============================================================================
            # Epigenomic, DNABERT and DNA curvature branches
            # Support switches: use_maskconv / use_gaussian
            # ==============================================================================
            def build_seq_branch(name, seq_len, input_dim, dim, dropout_rate, use_maskconv, use_gaussian):
                feat_input = layers.Input(shape=(seq_len, input_dim), name=f"{name}_feature")
                mask_input = layers.Input(shape=(seq_len,), name=f"{name}_valid_mask")

                if use_maskconv:
                    x = MaskConv1D(filters=48, kernel_size=7, padding="same", activation="relu")([feat_input, mask_input])
                    x = MaskConv1D(filters=48, kernel_size=5, padding="same", activation="relu")([x, mask_input])
                else:
                    # standard Conv1D, ignore mask
                    x = layers.Conv1D(48, kernel_size=7, padding="same", activation="relu")(feat_input)
                    x = layers.Conv1D(48, kernel_size=5, padding="same", activation="relu")(x)

                if use_gaussian:
                    pos = tf.range(seq_len, dtype=tf.float32)
                    center = seq_len / 2.0
                    sigma = seq_len / 6.0
                    gauss_weight = tf.exp(-tf.square(pos - center) / (2 * tf.square(sigma)))
                    gauss_weight = tf.expand_dims(gauss_weight, axis=0)
                    gauss_weight = tf.expand_dims(gauss_weight, axis=-1)
                    mask_2d = tf.expand_dims(mask_input, axis=-1)
                    weight = mask_2d * gauss_weight
                    weight = weight / (tf.reduce_sum(weight, axis=1, keepdims=True) + 1e-8)
                    x_weighted = x * weight
                else:
                    x_weighted = x

                x_max = layers.GlobalMaxPooling1D()(x_weighted)
                x_avg = layers.GlobalAveragePooling1D()(x_weighted)
                pooled = layers.Concatenate()([x_max, x_avg])
                x = layers.Dense(64, activation="relu")(pooled)
                x = layers.Dropout(dropout_rate)(x)
                feat_out = layers.Dense(dim, activation="relu")(x)
                feat_out = layers.LayerNormalization()(feat_out)
                return tf.keras.Model([feat_input, mask_input], feat_out)


            # ==============================================================================
            # Adaptive Fusion
            # ==============================================================================
            def fuse_multi_feat(feature_list, valid_scalars_list, dim=128, l2_main=2e-4, l2_score=1e-5,
                                dropout_rate=GLOBAL_DROPOUT):
                for f in feature_list:
                    assert f.shape[-1] == dim, f"All modality branch outputs must have dimension {dim}"
                num_modal = len(feature_list)
                if num_modal == 1:
                    feat = feature_list[0]
                    x = layers.Dense(512, activation="relu", kernel_regularizer=L2(l2_main))(feat)
                    x = layers.Dropout(dropout_rate)(x)
                    x = layers.Dense(dim, kernel_regularizer=L2(l2_main))(x)
                    out = layers.LayerNormalization(epsilon=1e-6)(x)
                    return out, None

                concat_all = layers.Concatenate(axis=-1)(feature_list)
                concat_all = layers.LayerNormalization(epsilon=1e-6)(concat_all)
                modal_score = layers.Dense(num_modal, kernel_regularizer=L2(l2_score))(concat_all)

                valid_stack = layers.Concatenate(axis=-1)(valid_scalars_list)
                penalty = (1.0 - valid_stack) * (-1e9)
                modal_score = layers.Add()([modal_score, penalty])

                alpha = layers.Softmax(axis=-1)(modal_score)

                weighted_feats = []
                for idx, feat in enumerate(feature_list):
                    weight = alpha[:, idx:idx + 1]
                    w_feat = layers.Multiply()([feat, weight])
                    weighted_feats.append(w_feat)
                fused = layers.Add()(weighted_feats)

                x = layers.Dense(512, activation="relu", kernel_regularizer=L2(l2_main))(fused)
                x = layers.Dropout(dropout_rate)(x)
                x = layers.Dense(dim, kernel_regularizer=L2(l2_main))(x)
                x = layers.Add()([x, fused])
                out = layers.LayerNormalization(epsilon=1e-6)(x)
                return out, alpha


            # ==============================================================================
            # Transformer Encoder
            # Code sourced from DeepTAD open-source repository
            # ==============================================================================
            def transformer_encoder(inputs, d_model, num_heads, mlp_dim, dropout_rate):
                x = layers.LayerNormalization(epsilon=1e-6)(inputs)
                x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)(x, x)
                x = layers.Add()([inputs, x])
                x = layers.LayerNormalization(epsilon=1e-6)(x)
                x = layers.Dense(mlp_dim, activation="gelu")(x)
                x = layers.Dropout(dropout_rate)(x)
                x = layers.Dense(d_model, activation="gelu")(x)
                x = layers.Dropout(dropout_rate)(x)
                x = layers.Add()([inputs, x])
                return x


            # ==============================================================================
            # Model initialization function
            # ==============================================================================
            def init_model(
                    use_epi=False,
                    use_dnabert=False,
                    use_curv=False,
                    use_transformer=False,
                    use_maskconv=True,
                    use_gaussian=True,
                    use_adaptive_fuse=True
            ):
                cnn_encoder = init_cnn()
                hic_in = cnn_encoder.input
                cnn_feat = cnn_encoder(hic_in)

                feat_collection = [cnn_feat]
                valid_collection = []
                input_collection = [hic_in]

                bsize = tf.shape(cnn_feat)[0]
                hic_valid = tf.ones((bsize, 1), dtype=tf.float32)
                valid_collection.append(hic_valid)

                if use_epi:
                    epi_encoder = build_seq_branch("epi", seq_len=100, input_dim=4, dim=128,
                                                   dropout_rate=GLOBAL_DROPOUT,
                                                   use_maskconv=use_maskconv, use_gaussian=use_gaussian)
                    epi_feat, epi_mask = epi_encoder.inputs
                    epi_out = epi_encoder([epi_feat, epi_mask])
                    feat_collection.append(epi_out)
                    input_collection.extend([epi_feat, epi_mask])
                    epi_scalar_valid = tf.expand_dims(tf.cast(tf.reduce_sum(epi_mask, axis=1) > 1e-6, tf.float32), axis=-1)
                    valid_collection.append(epi_scalar_valid)

                if use_dnabert:
                    dna_encoder = build_seq_branch("dnabert", seq_len=100, input_dim=64, dim=128,
                                                   dropout_rate=GLOBAL_DROPOUT,
                                                   use_maskconv=use_maskconv, use_gaussian=use_gaussian)
                    dna_feat, dna_mask = dna_encoder.inputs
                    dna_out = dna_encoder([dna_feat, dna_mask])
                    feat_collection.append(dna_out)
                    input_collection.extend([dna_feat, dna_mask])
                    dna_scalar_valid = tf.expand_dims(tf.cast(tf.reduce_sum(dna_mask, axis=1) > 1e-6, tf.float32), axis=-1)
                    valid_collection.append(dna_scalar_valid)

                if use_curv:
                    curv_encoder = build_seq_branch("curv", seq_len=100, input_dim=1, dim=128,
                                                    dropout_rate=GLOBAL_DROPOUT,
                                                    use_maskconv=use_maskconv, use_gaussian=use_gaussian)
                    curv_feat, curv_mask = curv_encoder.inputs
                    curv_out = curv_encoder([curv_feat, curv_mask])
                    feat_collection.append(curv_out)
                    input_collection.extend([curv_feat, curv_mask])
                    curv_scalar_valid = tf.expand_dims(tf.cast(tf.reduce_sum(curv_mask, axis=1) > 1e-6, tf.float32),
                                                       axis=-1)
                    valid_collection.append(curv_scalar_valid)

                if use_adaptive_fuse:
                    fused_feat, _ = fuse_multi_feat(feat_collection, valid_collection, dim=128)
                else:
                    # Simple feature concatenation and fusion
                    concat = layers.Concatenate(axis=-1)(feat_collection)
                    x = layers.Dense(512, activation="relu", kernel_regularizer=L2(2e-4))(concat)
                    x = layers.Dropout(GLOBAL_DROPOUT)(x)
                    x = layers.Dense(128, kernel_regularizer=L2(2e-4))(x)
                    fused_feat = layers.LayerNormalization(epsilon=1e-6)(x)

                if use_transformer:
                    fused_3d = layers.Reshape((1, 128))(fused_feat)
                    encoded = transformer_encoder(fused_3d, d_model=128, num_heads=4, mlp_dim=512,
                                                  dropout_rate=GLOBAL_DROPOUT)
                    encoded_pool = layers.GlobalAveragePooling1D()(encoded)
                else:
                    encoded_pool = fused_feat

                hidden = layers.Dense(128, activation="relu")(encoded_pool)
                hidden = layers.Dropout(GLOBAL_DROPOUT)(hidden)
                hidden = layers.Dense(64, activation="relu")(hidden)
                hidden = layers.Dropout(GLOBAL_DROPOUT)(hidden)
                outputs = layers.Dense(1, activation='sigmoid', kernel_regularizer=L2(1e-4))(hidden)

                model = tf.keras.Model(inputs=input_collection, outputs=outputs)
                model.compile(
                    loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=FOCAL_GAMMA, label_smoothing=0.02),
                    optimizer=keras.optimizers.Adam(learning_rate=INIT_LR, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                    metrics=[
                        keras.metrics.BinaryAccuracy(name='acc'),
                        keras.metrics.Precision(name='precision'),
                        keras.metrics.Recall(name='recall'),
                        F1Score(name="f1"),
                        keras.metrics.TruePositives(name='true_positives'),
                        keras.metrics.FalsePositives(name='false_positives'),
                        keras.metrics.TrueNegatives(name='true_negatives'),
                        keras.metrics.FalseNegatives(name='false_negatives')
                    ]
                )
                return model


            # ==============================================================================
            # Preprocessing function
            # ==============================================================================
            def fun(a):
                oshape = a.shape
                a = a.reshape(-1, 10).astype('float32')
                mean = np.mean(a, axis=1).reshape(-1, 1)
                a = a - mean
                sqrt = (np.sqrt(a.var(axis=1)) + 1e-10).reshape(-1, 1)
                a = a / sqrt
                return a.reshape(oshape)

            # Training Entry
            if __name__ == '__main__':
                # training set
                print("===== training set: Hi-C =====")
                train_data = np.load(
                    f"{root_dir}/{cell}/sample/{display_reso}kb/{ds_tag}/samples-generate/train/{cell}_{display_reso}kb_train_{ds_tag}_seed{SEED}.npy")
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
                    print(f"epi_train_feat shape:{epi_train_feat.shape}, mask:{epi_train_mask.shape}")
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
                    hic_path = f"{root_dir}/{cell}/sample/{display_reso}kb/{ds_tag}/samples-generate/validation/{cell}_{display_reso}kb_{chrom}_{ds_tag}.npy"
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
                    epi_val_feat = epi_val_feat[:,:,:4]
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
                    print("===== validation set: DNA curvature features =====")
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

                weight_out = f"{root_dir}/{cell}/sample/{display_reso}kb/{ds_tag}/weights"
                os.makedirs(weight_out, exist_ok=True)
                save_best_path = f"{weight_out}/{model_use}_best.h5"
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
                    batch_size=batch_size,
                    callbacks=callbacks,
                    shuffle=True
                )
                # Save training history
                hist_out = f"{root_dir}/{cell}/sample/{display_reso}kb/{ds_tag}/hist"
                os.makedirs(hist_out, exist_ok=True)
                np.save(f"{hist_out}/hist_{model_use}.npy", hist.history)


                # %%
                eval_val = model.evaluate(val_x, y_val, verbose=0)
                eval_train = model.evaluate(train_x, y_train, verbose=0)

                _, val_acc, val_prec, val_recall, val_f1, *_ = eval_val
                _, tr_acc, tr_prec, tr_recall, tr_f1, *_ = eval_train

                total_epoch = len(hist.history["val_f1"])
                if total_epoch < 100:
                    store_epoch = total_epoch - 15
                else:
                    store_epoch = total_epoch
                    print("\n" + "="*60)
                    print("⚠️ WARNING: 训练到达最大epoch=100，早停未触发！")
                    print("="*60 + "\n")

                log_csv = f"{root_dir}/{cell}/sample/run_log_{display_reso}kb.csv"
                row_data = [
                    model_use,
                    ds_tag,
                    resolution,
                    val_prec, val_recall, val_f1,
                    tr_prec, tr_recall, tr_f1,
                    store_epoch
                ]
                header = [
                    "model_use",
                    "ds_tag",
                    "resolution",
                    "val_prec","val_recall","val_f1",
                    "train_prec","train_recall","train_f1",
                    "store_epoch"
                ]

                file_exists = os.path.exists(log_csv)
                with open(log_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(header)
                    writer.writerow(row_data)

                print(f"\n====本次实验结果已写入 run_log_{display_reso}kb.csv====")
                print(f"res={resolution}, model_use={model_use}")
                print(f"【val】prec={val_prec:.4f}, recall={val_recall:.4f}, f1={val_f1:.4f}")
                print(f"【train】prec={tr_prec:.4f}, recall={tr_recall:.4f}, f1={tr_f1:.4f}")
                print(f"实际训练总轮数={total_epoch}, csv存储store_epoch={store_epoch}")
