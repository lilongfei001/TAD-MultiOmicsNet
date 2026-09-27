# Model ablation config switches
"""
use_epi:         Epigenetic features
use_dnabert:     DNABERT sequence features
use_curv:        DNA curvature features
use_transformer: Transformer block
use_maskconv:    MaskConv module
use_gaussian:    Gaussian weighting module
use_adaptive_fuse: Adaptive fusion module
"""
model_cfg = {
    "use_epi": True,
    "use_dnabert": True,
    "use_curv": True,
    "use_transformer": False,
    "use_maskconv": True,
    "use_gaussian": True,
    "use_adaptive_fuse": True,
}

def get_model_name(seed):
    name_parts = []
    if model_cfg["use_transformer"]: name_parts.append("Tr")
    if model_cfg["use_epi"]: name_parts.append("Ep")
    if model_cfg["use_dnabert"]: name_parts.append("Dn")
    if model_cfg["use_curv"]: name_parts.append("Cv")
    if model_cfg["use_maskconv"]: name_parts.append("Mk")
    if model_cfg["use_gaussian"]: name_parts.append("Gs")
    if model_cfg["use_adaptive_fuse"]: name_parts.append("Af")
    return f"HiC_{'_'.join(name_parts)}_seed{seed}"
