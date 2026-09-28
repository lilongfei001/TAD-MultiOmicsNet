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
