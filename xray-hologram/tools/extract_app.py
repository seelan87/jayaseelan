"""Extract verified source from the compressed notebook for Windows packaging."""
from pathlib import Path
import ast,base64,hashlib,json,sys,zlib

project=Path(__file__).resolve().parents[1]
original=Path(sys.argv[1])
book=json.loads(original.read_text(encoding='utf-8'))
assert len(book['cells'])==1 and book['cells'][0]['cell_type']=='code'
cell=book['cells'][0]['source'];cell=''.join(cell) if isinstance(cell,list) else cell
tree=ast.parse(cell)
payload=next(node for node in tree.body if isinstance(node,ast.Assign) and any(
    isinstance(target,ast.Name) and target.id=='_holo_payload' for target in node.targets))
source=zlib.decompress(base64.b64decode(ast.literal_eval(payload.value))).decode('utf-8')
original_digest=hashlib.sha256(source.encode()).hexdigest()
assert original_digest==book['metadata']['finalized_runtime']['sha256'], 'Compressed notebook payload checksum failed'
assert original_digest=='3429f137a8eb353892f6c3ad6e3c7c171c49cab9c6a39468c58b7f548a255ecf'
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
module=project/'src/xray_app/core.py';module.parent.mkdir(parents=True,exist_ok=True)
module.write_text(source,encoding='utf-8')
provenance=dict(notebook=original.name,notebook_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
    source_format='compressed single-cell notebook; statically decoded, verified payload',source_sha256=original_digest,
    packaged_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    adaptations=['Launcher owns Tk mainloop','Discover bundled CUDA DLLs and headers'],tabs=15)
(project/'source-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
print(json.dumps(provenance,indent=2))
