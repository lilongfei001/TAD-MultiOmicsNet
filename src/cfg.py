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
    if model_cfg["use_transformer"]:name_parts.append("Tr")
    if model_cfg["use_epi"]:name_parts.append("Ep")
    if model_cfg["use_dnabert"]:name_parts.append("Dn")
    if model_cfg["use_curv"]:name_parts.append("Cv")

    flag_ep_dn_cv_all_off = (not model_cfg["use_epi"]) and (not model_cfg["use_dnabert"]) and (not model_cfg["use_curv"])

    if not flag_ep_dn_cv_all_off:
        if model_cfg["use_maskconv"]:name_parts.append("Mk")
        if model_cfg["use_gaussian"]:name_parts.append("Gs")
        if model_cfg["use_adaptive_fuse"]:name_parts.append("Af")

    parts_str = "_".join(name_parts)
    if parts_str:
        return f"HiC_{parts_str}_seed{seed}"
    else:
        return f"HiC_seed{seed}"
