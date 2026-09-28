# Epigenomic signal aggregation Workflow
- Tool: deepTools
- Version: 3.5.6
- Repo: https://github.com/deeptools/deeptools

- CTCF, H3k4me3, H3k27ac, H3k27me3
- cell: GM12878,K562
- 
multiBigwigSummary BED-file --BED all_1kb_bins.bed -b CTCF.bigWig H3k4me3.bigWig H3k27ac.bigWig H3k27me3.bigWig -o epi4.npz --outRawCounts epi4.tab -p 8 --smartLabels

Note:
all_1kb_bins.bed: genome-wide fixed equal-width bin BED file matching Hi-C bin coordinates, generated from UCSC hg19 hg19.chrom.sizes.
