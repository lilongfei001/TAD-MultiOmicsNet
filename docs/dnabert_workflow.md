# DNABERT feature extraction Workflow
- Script path: docs/dnabert_extract_feat.py
- ⚠️ Environment note: This script requires a **separate Python environment**, different from the model training environment. GPU is required for efficient inference.
- Tested on AutoDL: PyTorch 2.0.0 + Python 3.8 + CUDA 11.8
- Package versions: torch==2.0.0, transformers==4.29.2, biopython, psutil, einops==0.6.1, numpy

## DNABERT1 model
DNABERT1 pre-trained weights: https://huggingface.co/zhihan1996/DNA_bert_6/tree/main

## Install dependencies (AutoDL example)
```bash
pip install transformers==4.29.2 biopython psutil einops==0.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

# Note:
- Input: genome_1kb.fasta (1kb-binned genome sequence), pre-trained DNABERT1 model weights.
- genome_1kb.fasta: The genome is split into non-overlapping 1kb bins according to Hi-C bin boundaries. Each fasta record corresponds to the DNA sequence within one 1kb genomic bin.
- Output: Block-wise saved .npy embeddings for each chromosome, plus merged global 1kb base feature matrices.
- SAVE_1KB_BASE stores genome-wide sorted 1kb DNABERT embeddings for each chromosome, merged from chunked intermediate block outputs, for convenient loading in subsequent model training and inference.
- Sliding window: 400bp window, 200bp stride; embedding averaged across windows within each bin.
- Failed sequences or bins with all N bases will return zero 768-dimensional vectors.
- Supports chunked processing to reduce memory usage; existing blocks will be skipped to enable resume.
