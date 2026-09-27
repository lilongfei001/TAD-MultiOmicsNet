## Usage
### 1. Training
- Modify ablation switches in `src/cfg.py` first, then run the training script: `src/model_train.py`

### 2. Prediction and Evaluation
- GM12878 cell line: run `predict_eval.ipynb`
- K562 cell line supplementary experiment: run `predict_eval_K562.ipynb`
- Missing modality ablation: run `predict_missing_ablation.ipynb`

### 3. Preprocessing
- Scripts in `src/preprocess/` generate multi-omics input data, including Hi-C matrix, epigenetic signals, DNA curvature and DNABERT sequence embeddings.

### 4. Analysis & Supplementary
- `src/alpha_analysis/`: Visualization of adaptive fusion weights
- `src/meta_profile_analysis/`: Meta-profile epigenetic enrichment analysis
- `src/K562_supplement/`: K562 cell line supplementary experiments, including K562 multi-omics data preprocessing.

### Ablation Configuration (src/cfg.py)
- Boolean switches in configuration file (`cfg.py`) are used to enable or disable individual modules for ablation studies.

