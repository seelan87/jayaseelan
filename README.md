# MuMax3 Image Tools — Windows x64

Four-tab Windows desktop app for OVF/OOMMF/VTK vector-field images, videos,
cross sections, 3D arrows/cuboids, streamlines, and isosurfaces. Includes shared
ParaView color presets, ROI/upscaling, layer fills/merge sets, lighting,
reflection/transparency, and PNG/MP4 export.

## Download and run

1. Open **Actions → Build Windows x64 EXE** and select the latest successful run.
2. Download the **MuMax3-Image-Tools-windows-x64** artifact and extract it.
3. Double-click **MuMax3-Image-Tools-windows-x64.exe** for the self-contained app.
   Python and Jupyter are not required. First launch extracts the bundled libraries
   into Windows' temporary folder and can take longer than later GUI operations.

For faster startup, extract **MuMax3-Image-Tools-windows-x64-portable.zip** and
run **MuMax3-Image-Tools.exe** inside its folder. Keep its `_internal` folder next
to the executable. Both distributions provide the same application features.

Target: Windows 10/11 **64-bit x64**. GPU rendering uses the installed graphics
driver. Tab 3 has an analytic CPU fallback; Tab 4's VTK rendering needs a working
OpenGL-capable graphics driver. The build does not include an NVIDIA CUDA runtime.

Native OVF/OOMMF/VTK reading and normal VTK conversion are included. Recovery of an
otherwise unreadable OVF may optionally use the official **mumax3-convert.exe**:
place it next to the app, in its `bin` folder, on PATH, or set `MUMAX3_CONVERT`.
This optional external utility is not redistributed in these downloads.

App diagnostics: `%LOCALAPPDATA%\MuMax3ImageTools\logs\app.log`.
VTK cache files are temporary and are removed when the app closes.

## Build using GitHub

The Windows x64 workflow runs on each push to `master`/`main`, on pull requests,
and through **Actions → Build Windows x64 EXE → Run workflow**. It installs pinned
dependencies with 64-bit Python 3.12, checks the source app, builds the portable
executable and tests it, then builds and tests the single-file executable.
Downloads include SHA-256 checksums and dependency license notices.

## Build locally

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\.venv\Scripts\python.exe tools\build_windows.py --mode both
```

Output: `release/`. To run the source app: `python launcher.py`.
To check it without leaving a GUI window open:
`python launcher.py --self-test test-results/source.json`.
The same `--self-test` option works on the executable. Add `--render-test` to
exercise GPU rendering on a computer with suitable graphics hardware.

## Source and third-party components

`src/mumax_app/core.py` contains the current notebook application's complete GUI
and analysis code, with an import-safe launcher and portable converter discovery.
The notebook remains available separately and is not required by this build.
ParaView preset notices and bundled dependency licenses are included in the build.
No project-wide open-source license has been selected for the application code.

Build references: [PyInstaller](https://pyinstaller.org/en/stable/usage.html),
[PyVista freezing guide](https://docs.pyvista.org/extras/pyinstaller.html).
