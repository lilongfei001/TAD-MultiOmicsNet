# Hi‑C Downsample
- Tool: cooltools
- Version: 0.7.1
- Repo:https://github.com/open2c/cooltools
> This document only covers the random downsampling step for raw Hi-C cool matrices.
> Steps including `.hic` to `.cool` conversion, ICE normalization and balanced `.cool` to `.hic` export are described in `docs/hic_ice_workflow.md`.

## Example Command
```bash
cooltools random-sample -f 0.5 -p 1 GM12878_10kbraw.cool sample_50pct_r1_10kb.cool

Command breakdown:
- GM12878_10kbraw.cool: Input raw-count Hi-C contact matrix at 10 kb resolution
- -f 0.5: Randomly retain 50% of the original contact pairs
- -p 1: Run the sampling task with 1 process
- sample_50pct_r1_10kb.cool: Output downsampled matrix
  - 50pct: Sampling fraction (50%)
  - r1: Replicate number 1 (r stands for replicate)
  - 10kb: Matrix resolution
Four downsampling fractions are used in this experiment: 0.1, 0.25, 0.5, 0.75, with 3 independent replicates for each fraction.
