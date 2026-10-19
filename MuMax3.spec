# PyInstaller specification for a Windows x64 Tk + Matplotlib + VTK application.
import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

project = Path(SPECPATH)
name = 'MuMax3-Image-Tools'
mode = os.environ.get('MUMAX_BUNDLE_MODE', 'onefile')
datas = [(str(project/'src/mumax_app/core.py'), 'mumax_app'),
         (str(project/'THIRD_PARTY_NOTICES.txt'), '.'),
         (str(project/'README.md'), '.')]
for package in ('pyvista', 'numba', 'llvmlite'):
    datas += collect_data_files(package, include_py_files=False)
for distribution in ('pyvista', 'vtk', 'numba', 'llvmlite', 'numpy', 'matplotlib', 'scipy', 'pillow', 'opencv-python'):
    datas += copy_metadata(distribution)
binaries = collect_dynamic_libs('vtkmodules') + collect_dynamic_libs('llvmlite')
hiddenimports = ['matplotlib.backends.backend_tkagg', 'matplotlib.backends.backend_agg',
                 'PIL._tkinter_finder', 'vtkmodules', 'vtkmodules.all',
                 'vtkmodules.util.numpy_support', 'vtkmodules.numpy_interface.dataset_adapter',
                 'numba', 'llvmlite.binding', 'scipy.spatial._ckdtree']
hiddenimports += collect_submodules('vtkmodules.numpy_interface')
hiddenimports += collect_submodules('vtkmodules.util')
a = Analysis([str(project/'launcher.py')], pathex=[str(project/'src')],
             binaries=binaries, datas=datas, hiddenimports=hiddenimports,
             hookspath=[], hooksconfig={'matplotlib': {'backends': ['TkAgg', 'Agg']}},
             runtime_hooks=[], excludes=['IPython','jupyter','notebook','jupyterlab',
                 'PyQt5','PyQt6','PySide2','PySide6','wx','pytest',
                 'numba.tests','llvmlite.tests','vtkmodules.qt'],
             noarchive=False)
# A venv based on Conda can inherit an unrelated Tcl/Tk DLL from another Python
# on PATH. Match the collected init.tcl/tk.tcl files to the base interpreter's
# DLLs. On standard CPython/GitHub runners these already point to the same files.
for dll in ('tcl86t.dll', 'tk86t.dll'):
    candidates = [Path(sys.base_prefix)/'Library/bin'/dll, Path(sys.base_prefix)/'DLLs'/dll]
    matching = next((path for path in candidates if path.is_file()), None)
    if matching is not None:
        a.binaries = [(dest, str(matching) if Path(dest).name.lower()==dll else src, kind)
                      for dest, src, kind in a.binaries]
pyz = PYZ(a.pure)
options = dict(name=name, debug=False, bootloader_ignore_signals=False,
               strip=False, upx=False, console=False, disable_windowed_traceback=False)
if mode == 'onedir':
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, **options)
    coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=name)
else:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], **options)
