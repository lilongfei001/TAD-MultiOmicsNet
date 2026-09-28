# Hi‑C ICE Workflow
Tool: hictk
Version: v2.2.0
Repo: https://github.com/paulsengroup/hictk

```bash
# hic → cool
hictk convert <input.hic> <out_file.cool> --resolutions <res>

# ICE normalization (cis-only)
hictk balance ice --mode cis --ignore-diags 1 --min-count 0 --min-nnz 10 --max-iters 1000 --tolerance 1e-05 -t 8 -f --in-memory <out_file.cool>

# cool → balanced hic
hictk convert -r <res> --normalization-methods ICE -t 16 -f -v 4 <out_file.cool> <balanced.hic>

Note: Replace placeholders <> with your own file names and resolution value (10000 or 25000).
ICE normalization runs in-place on the cool file. Balanced hic matrices are used for downstream TAD calling and feature preprocessing.
