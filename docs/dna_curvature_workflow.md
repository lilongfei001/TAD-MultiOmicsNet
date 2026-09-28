# DNA curvature calculation Workflow
- Script path: docs/calc_dna_curvature.py
- ⚠️ Environment note: This script requires a **separate Python environment**, different from the model training environment.
- Package versions: dnacurve==2026.6.9, pyfaidx, numpy, matplotlib

## Create conda environment
```bash
conda create -n dnacurve_env python=3.13
conda activate dnacurve_env
conda install pip setuptools
python -m pip install dnacurve pyfaidx numpy matplotlib
```

# Note:
Input: hg19 reference genome FASTA, all_1kb_bins.bed (1kb equal-width bins generated from UCSC hg19 chrom.sizes).
Output CSV: columns chr, start, end, curv_mean, mean DNA curvature per bin.
Built-in resume function: can continue interrupted computation from last saved record.
Bins with >90% N bases are marked as NaN and excluded.
