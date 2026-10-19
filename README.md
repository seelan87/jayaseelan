# MuMax3 Image Tools — Windows x64

Four-tab Windows desktop app for OVF/OOMMF/VTK vector-field images, videos,
cross sections, 3D arrows/cuboids, streamlines, and isosurfaces. Includes shared
ParaView color presets, ROI/upscaling, layer fills/merge sets, lighting,
reflection/transparency, and PNG/MP4 export.

Tabs 3 and 4 include **Layer Fill → Finish (3D) → Glass** for translucent layers
with sharp white highlights. Use **Transparency (%)** to tune the selected layer
or merge set. Choosing Glass on an opaque layer starts at 70% transparency;
existing transparency settings are retained. This surface finish does not trace
optical refraction.

Tabs 1, 3 and 4 use a single searchable **ParaView color preset** gallery.
Tab 4's always-visible **3D view** selector offers 3D, XY, YZ and XZ camera
orientations independently of the cross-section controls. In Tabs 3 and 4,
**Layer Fill → Edge enable** outlines the selected layer or merge set, preserving
holes and the outer shape without drawing internal cell or merged-layer seams.
Tab 4's Display panel includes **Background color**, **Show outline**, and
**Outline color**. PNG and MP4 exports retain these settings; labels adapt to
light and dark backgrounds.
The shared gallery includes exact RGB **Red → White → Blue** and
**Blue → White → Red** ramps. The selected preset drives both field colors and
the corresponding color bar in views and exports.

## Download and run

1. Open [the latest Windows release](https://github.com/seelan87/jayaseelan/releases/latest).
2. Download **MuMax3-Image-Tools-windows-x64.exe** from its assets.
3. Double-click it for the self-contained app.
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
Successful builds from the default branch are also published under **Releases**.
The complete build and test reports remain available as GitHub Actions artifacts.

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
