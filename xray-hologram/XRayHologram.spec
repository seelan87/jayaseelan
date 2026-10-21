import os,sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files,collect_dynamic_libs,collect_submodules,copy_metadata

project=Path(SPECPATH)
name='XRay-Hologram-Reconstruction'
mode=os.environ.get('XRAY_BUNDLE_MODE','onedir')
datas=[(str(project/'README.md'),'.'),(str(project/'THIRD_PARTY_NOTICES.txt'),'.'),
       (str(project/'source-provenance.json'),'.')]
for distribution in ('numpy','pandas','h5py','scipy','scikit-image','matplotlib','Pillow',
                     'imageio','imageio-ffmpeg','cupy-cuda12x','fastrlock',
                     'nvidia-cuda-runtime-cu12','nvidia-cuda-nvrtc-cu12','nvidia-cufft-cu12','nvidia-nvjitlink-cu12'):
    datas+=copy_metadata(distribution)
datas+=collect_data_files('imageio_ffmpeg',include_py_files=False)
datas+=collect_data_files('cupy',include_py_files=False)
datas+=collect_data_files('cupyx',include_py_files=False)
# Preserve NVIDIA wheel layout: core finds DLLs and NVRTC headers relative to
# _MEIPASS/nvidia, with CPU fallback on computers without an NVIDIA driver.
for library in ('cuda_runtime','cuda_nvrtc','cufft','nvjitlink'):
    root=Path(sys.prefix)/'Lib/site-packages/nvidia'/library
    if not root.exists():raise RuntimeError('Missing CUDA runtime wheel: '+str(root))
    datas.append((str(root),'nvidia/'+library))
binaries=[]
zlib=Path(sys.base_prefix)/'Library/bin/zlib1.dll'
if zlib.is_file():binaries.append((str(zlib),'.'))
hiddenimports=['matplotlib.backends.backend_tkagg','matplotlib.backends.backend_agg',
    'PIL._tkinter_finder','tkinter.simpledialog','imageio.plugins.ffmpeg','imageio_ffmpeg',
    'scipy.spatial._ckdtree','skimage.feature','skimage.segmentation',
    'cupy','cupy.cuda','cupy.cuda.nvrtc','cupy.cuda.cufft','cupy.fft','fastrlock','fastrlock.rlock']
hiddenimports+=collect_submodules('cupy_backends')
hiddenimports+=collect_submodules('cupy',filter=lambda name:'.tests' not in name)
a=Analysis([str(project/'launcher.py')],pathex=[str(project/'src')],datas=datas,
    binaries=binaries,hiddenimports=hiddenimports,
    hooksconfig={'matplotlib':{'backends':['TkAgg','Agg']}},
    excludes=['IPython','jupyter','notebook','jupyterlab','PyQt5','PyQt6',
              'PySide2','PySide6','wx','pytest','torch','tensorflow','cv2'],noarchive=False)
for dll in ('tcl86t.dll','tk86t.dll'):
    candidates=[Path(sys.base_prefix)/'Library/bin'/dll,Path(sys.base_prefix)/'DLLs'/dll]
    matching=next((p for p in candidates if p.is_file()),None)
    if matching:
        a.binaries=[(dest,str(matching) if Path(dest).name.lower()==dll else src,kind)
                    for dest,src,kind in a.binaries]
pyz=PYZ(a.pure)
options=dict(name=name,console=False,debug=False,strip=False,upx=False,
             disable_windowed_traceback=False)
if mode=='onedir':
    exe=EXE(pyz,a.scripts,[],exclude_binaries=True,**options)
    coll=COLLECT(exe,a.binaries,a.datas,strip=False,upx=False,name=name)
else:
    exe=EXE(pyz,a.scripts,a.binaries,a.datas,[],**options)
