# Third-party notices

Code in this package is under the MIT License. Third-party projects keep their own licenses.

## TDL-GNSS

- Repository: <https://github.com/ebhrz/TDL-GNSS>
- Pinned commit: `a640b2832c90daeb1ce25644a1da91c8edeb13fc`
- Used component: the `HybridShareSysNet` layout, `9→64→128→64→2`, and the weight-and-correction output.
- Results here are an adaptation to the shared data and positioning backend. They are not a rerun of the upstream `train.py` or of that paper's original data.

## GNC-and-ADAPT

- Repository: <https://github.com/MIT-SPARK/GNC-and-ADAPT>
- Pinned commit: `dd5fe1f51839a8a43782fc54f0ba9aff24f5402f`
- License: BSD 2-Clause.
- This package adapts the GNC-TLS weight update to GNSS. It is not an official upstream GNSS result.

## gnss_lib_py

- Repository: <https://github.com/Stanford-NavLab/gnss_lib_py>
- License: MIT.
- The residual FDE path uses the same residual test idea on a shared multi-constellation, physically whitened, QR-projected backend. It is not a numerical copy of the Stanford pipeline.
