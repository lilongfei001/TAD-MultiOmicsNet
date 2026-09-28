import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
from collections import defaultdict

warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# =====================【请修改这里】=====================
MODEL_PATH = "/root/autodl-tmp/dnabert1"
BIN_FASTA = "/root/autodl-tmp/genome_1kb.fasta"
chr_list = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
SAVE_FEAT_ROOT = "/root/autodl-tmp/chr_1kb_dna_feats"
SAVE_1KB_BASE = "/root/autodl-tmp/chr_1kb_dna_base"
# =======================================================
BATCH_SIZE = 1000
# 原500 → 改为10000，分块更大，减少文件数量
CHR_BLOCK_SIZE = 40000
WINDOW_BATCH = 48
KB_STEP = 1000  # 新增1kb常量

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
os.makedirs(SAVE_FEAT_ROOT, exist_ok=True)
os.makedirs(SAVE_1KB_BASE, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True).to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False


def single_dna_emb(dna_seq: str):
    seq_upper = dna_seq.strip().upper()
    total_len = len(seq_upper)
    if total_len == 0 or seq_upper.count("N") == total_len:
        return np.zeros(768, dtype=np.float32)

    window_bp = 400
    stride_bp = 200
    window_seqs = []
    start_idx = 0
    while start_idx < total_len:
        sub_dna = seq_upper[start_idx: start_idx + window_bp]
        window_seqs.append(sub_dna)
        start_idx += stride_bp

    window_embeds = []
    for i in range(0, len(window_seqs), WINDOW_BATCH):
        batch_seqs = window_seqs[i:i + WINDOW_BATCH]
        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)
        batch_vecs = out[0].mean(dim=1).cpu().numpy()
        window_embeds.extend(batch_vecs)

    full_bin_vec = np.mean(np.array(window_embeds), axis=0)
    return full_bin_vec


def extract_all_bin_features_chr(chr_records, chr_name):
    chr_out_dir = os.path.join(SAVE_FEAT_ROOT, chr_name)
    os.makedirs(chr_out_dir, exist_ok=True)
    total = len(chr_records)
    print(f"开始处理 {chr_name}，共{total}个bin，块大小：{CHR_BLOCK_SIZE} | 仅bin本体序列，无侧翼")

    # 全局缓存：保存整条染色体所有1kb向量+坐标
    full_chr_feats = []
    full_chr_kbidx = []

    block_idx = 0
    for start_pos in range(0, total, CHR_BLOCK_SIZE):
        end_pos = start_pos + CHR_BLOCK_SIZE
        block_records = chr_records[start_pos:end_pos]
        feat_path = os.path.join(chr_out_dir, f"{chr_name}_block_{block_idx}_feat.npy")
        name_path = os.path.join(chr_out_dir, f"{chr_name}_block_{block_idx}_names.npy")
        if os.path.exists(feat_path) and os.path.exists(name_path):
            print(f"  {chr_name} block{block_idx} 已存在，跳过 [{start_pos}:{end_pos})")
            # 读取已存在块，补全全局缓存
            block_feat = np.load(feat_path)
            block_names = np.load(name_path, allow_pickle=True)
            for name, vec in zip(block_names, block_feat):
                s_bp = int(name.split(":")[1].split("-")[0])
                kb_idx = s_bp // KB_STEP
                full_chr_feats.append(vec)
                full_chr_kbidx.append(kb_idx)
            block_idx += 1
            continue

        feat_list = []
        bin_names = []
        print(f"  正在计算 {chr_name} block{block_idx} [{start_pos}:{end_pos})")
        for off, rec in enumerate(block_records):
            global_idx = start_pos + off + 1
            bin_id = rec.id
            try:
                dna_input = str(rec.seq)
                emb = single_dna_emb(dna_input)
            except Exception as e:
                print(f"  !警告 {chr_name} {bin_id} 提取失败，使用零向量, err: {str(e)}")
                emb = np.zeros(768, dtype=np.float32)
            feat_list.append(emb)
            bin_names.append(bin_id)
            # 存入全局完整缓存
            s_bp = int(bin_id.split(":")[1].split("-")[0])
            kb_idx = s_bp // KB_STEP
            full_chr_feats.append(emb)
            full_chr_kbidx.append(kb_idx)

            if global_idx % BATCH_SIZE == 0:
                print(f"  {chr_name} 已完成 {global_idx}/{total}")
        block_feats = np.array(feat_list, dtype=np.float32)
        np.save(feat_path, block_feats)
        np.save(name_path, np.array(bin_names, dtype=object))
        print(f"  {chr_name} block{block_idx} 完成，shape: {block_feats.shape}")
        block_idx += 1

    # 全部block跑完后，排序并保存整条染色体1kb基底（关键新增）
    print(f"\n合并{chr_name}全部1kb特征，生成全局基底文件")
    # 按基因组kb索引升序
    combine = list(zip(full_chr_kbidx, full_chr_feats))
    combine.sort(key=lambda x: x[0])
    sorted_kb_idx = [x[0] for x in combine]
    sorted_feats = [x[1] for x in combine]

    mat = np.array(sorted_feats, dtype=np.float32)
    feat_save = os.path.join(SAVE_1KB_BASE, f"{chr_name}_all_1kb_feat.npy")
    idx_save = os.path.join(SAVE_1KB_BASE, f"{chr_name}_1kb_index_map.npy")
    np.save(feat_save, mat)
    np.save(idx_save, np.array(sorted_kb_idx))
    print(f"{chr_name} 全局1kb基底保存完成，shape={mat.shape}\n")


if __name__ == "__main__":
    test_seq = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
    test_vec = single_dna_emb(test_seq)
    print("测试向量维度：", test_vec.shape)
    chr_group = defaultdict(list)
    for rec in SeqIO.parse(BIN_FASTA, "fasta"):
        chr_name = rec.id.split(":")[0]
        chr_group[chr_name].append(rec)
    chr_group = {k: v for k, v in chr_group.items() if k in chr_list}
    print("待处理染色体：", list(chr_group.keys()))
    for chr_name, rec_list in chr_group.items():
        extract_all_bin_features_chr(rec_list, chr_name)
    print("✅ DNA特征提取全部完成！")