# X-Ray Hologram Reconstruction — Windows 11 x64

The 15-tab hologram analysis application from `finalized_2d_analysis.ipynb`, packaged
for 64-bit Windows. Python, Jupyter, CUDA Toolkit, and FFmpeg installations are
not required. The notebook's scientific fitting and processed ROI workflows
are preserved.

## Run

- Double-click `XRay-Hologram-Reconstruction-windows-x64.exe` for the single-file app.
- For faster startup, extract the portable ZIP and run
  `XRay-Hologram-Reconstruction.exe` inside its folder. Keep `_internal` beside it.

The single-file app extracts its bundled libraries to a temporary directory at
startup, so it takes longer to start than the portable build. All 15 tabs appear
together after the loading screen. This build is unsigned; Windows may show the
publisher as unknown.

Large FFTs automatically benchmark the bundled GPU backend against CPU processing.
An NVIDIA CUDA-compatible driver is needed for GPU execution. Machines without a
compatible GPU automatically use the CPU. GPU runtime libraries are bundled;
CUDA Toolkit installation is unnecessary. GUI controls and scientific fitting
remain on the CPU.

Logs: `%LOCALAPPDATA%\XRayHologram\logs\app.log`.
User datasets and calibration JSON files are selected through the GUI and are
not included in the application or uploaded by the build workflow.

## Features retained

- Calibration **Fix** survives **New Upload**; default branch is +4 A to −4 A.
- Tab 10 **Fix Center** tracks the core between frames.
- Imaginary-component and processed ROI analysis, including matched Tab 10/11 fits.
- Zero-based frame sliders with midpoint initial positions.
- Home, Back/Forward, Pan, Zoom, subplot settings, Save, and MP4 export.
- Raw and normalized selected-component intensity modes, with matching colorbar scales.
- Tab 6 starts and resets with Filter **None**; other filters are available on demand.
- Direct 2D signed full-ROI intensity, local-background subtraction, central-core RMS
  diameter, and minimum outer circularity filtering in Tabs 10 and 11.
- Cached per-frame figures after tracking: slider labels update immediately;
  dragging performs no calculation and release displays the stored result.
- Restored 1D Gaussian/residual plots after leaving 2D mode, combined 1D/2D
  plot-axis menus, Show Legends controls, and Tab 6 FFT/IFFT picture-in-picture.

## Build using GitHub Actions

Run **Build X-Ray Hologram Windows x64** from the repository's Actions page.
It uses Windows and Python 3.12 x64, installs pinned dependencies, tests the
source app, builds both formats, and tests each packaged executable, including
the 2D shape filter and calculation-free slider browsing. The
`XRay-Hologram-Windows-x64` artifact contains both downloads, checksums, source
provenance, and self-test reports. No release or existing application is replaced.

## Build locally

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\.venv\Scripts\python.exe launcher.py --self-test test-results/source.json
.\.venv\Scripts\python.exe tools/build_windows.py --mode both
```

Full readable application source: `src/xray_app/core.py`.

## Notebook formats and measurement exports

The download includes `finalized_2d_analysis_readable.ipynb` and
`finalized_2d_analysis_compressed.ipynb`, both in a single cell, plus readable
Python source. Both notebooks contain the same verified application source.
The compressed notebook reduces Jupyter startup overhead; the readable notebook
shows the complete editable code directly.

Tab 1 offers **EXPORT ALL MEASUREMENT TABLES**. It snapshots stored results,
calibration, ROI geometry and finalized processing settings without running
analysis. `Export_Manifest.csv` identifies exported and unavailable tables;
unresolved values remain unavailable rather than fabricated zeros.

Tab 7 table export includes full-precision Gaussian and endpoint companions.
Tab 8 table and graph exports include rectangular per-frame, per-particle,
D-fit particle and all-frame radial-edge CSVs. Tab 11 displays complete scalar
fit results, covariance, diagnostics and integrated-intensity uncertainties.
Frame_Index is zero-based; Frame is one-based. Size is nm, area is nm^2,
1D integrated intensity is a.u. nm, and 2D integrated intensity is a.u. nm^2.
Intensity provenance distinguishes raw reconstruction from normalized signal.

The GitHub build tests actual export/readback paths before and after freezing.
No user datasets are included in the repository or application downloads.

The EXE source is extracted from the compressed notebook payload and checksum-verified before packaging. To prepare a source checkout, run the notebook builder and then tools/extract_app.py on release/finalized_2d_analysis_compressed.ipynb before running launcher.py.
