"""Create the packaged module from the reviewed notebook's readable source."""
from pathlib import Path
import ast,hashlib,json,sys

project=Path(__file__).resolve().parents[1]
original=Path(sys.argv[1])
source=original.read_text(encoding='utf-8')
original_digest=hashlib.sha256(source.encode()).hexdigest()
assert original_digest=='4ea1bee46319cea4aa41929c733110525fb52348ba58c11e3f99e21227700fe8'
assert source.count('_gui_event_loop_mode = _run_gui_event_loop(root)')==1
source=source.replace('_gui_event_loop_mode = _run_gui_event_loop(root)',
    "_gui_event_loop_mode = 'standalone-ready'",1)
old="runtime_roots = [cache/'nvidia'] + [p/'nvidia' for p in cache.parent.glob('cp*') if p != cache]"
new="""runtime_roots = [Path(getattr(sys, '_MEIPASS', sys.prefix))/'nvidia',
                             Path(sys.prefix)/'Lib'/'site-packages'/'nvidia', cache/'nvidia']
            runtime_roots += [p/'nvidia' for p in cache.parent.glob('cp*') if p != cache]"""
assert source.count(old)==1
source=source.replace(old,new,1)
source=source.replace("for library in ('cuda_runtime', 'cuda_nvrtc', 'cufft'):",
                      "for library in ('cuda_runtime', 'cuda_nvrtc', 'cufft', 'nvjitlink'):",1)
compile(source,'xray_app/core.py','exec')
(project/'src/xray_app/core.py').write_text(source,encoding='utf-8')
provenance={'notebook':'finalized_2d_analysis.ipynb','source_sha256':original_digest,
    'packaged_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
    'adaptations':['Launcher owns Tk mainloop','Discover bundled CUDA DLLs and headers'],
    'platform':'Windows 11 x64','tabs':15}
(project/'source-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(provenance,indent=2))
