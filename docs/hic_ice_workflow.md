# Hi‑C ICE Workflow
Tool: hictk

```bash
# hic → cool
hictk convert <input.hic> <out.cool> --resolutions 10000,25000

# ICE normalization (cis-only)
hictk balance ice --mode cis --ignore-diags 1 --min-count 0 --min-nnz 10 --max-iters 1000 --tolerance 1e-05 -t 8 -f --in-memory <in.cool> <out_ice.cool>

# cool → balanced hic
hictk convert -r <res> --normalization-methods ICE -t 16 -f -v 4 <ice.cool> <balanced.hic>
