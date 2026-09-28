All raw TAD calling used hictk-generated ICE-normalized Hi-C matrices: stored as .hic/.cool, or exported from .hic to text files (only format converted, contact values unchanged).

### TAD Caller Details
1. **deDoc**
> GitHub: https://github.com/yinxc/structural-information-minimisation
> Parameters: -E; remaining arguments kept default.
> Note: Uses 1-based bin indices; bins converted to bp by `start=(start_bin-1)*resolution`, `end=end_bin*resolution`.

2. **TopDom (R package v0.10.1)**
> GitHub: https://github.com/HenrikBengtsson/TopDom
> Parameters: window.size=5; remaining arguments kept default.
> Note: Only "domain" regions were retained.

3. **Arrowhead (Juicer v2.20.00)**
> GitHub: https://github.com/aidenlab/Juicebox/releases
> Parameters: -k ICE; remaining arguments kept default.

4. **SpectralTAD (R package v1.26.0)**
> GitHub: https://github.com/dozmorovlab/SpectralTAD
> Parameters: levels=2, z_clust=TRUE, window_size=80(25kb)/200(10kb), resolution=25000/10000; remaining arguments kept default.

5. **Domaincaller (DI, domaincaller v0.1.0)**
> GitHub: https://github.com/XiaoTaoWang/domaincaller
> Parameters: --weight-col ICE; remaining arguments kept default.

## Post-processing
After raw TAD calling, all boundary coordinates were rounded to the nearest multiple of Hi-C bin resolution (10 kb / 25 kb), to align boundaries with Hi-C bin edges.
