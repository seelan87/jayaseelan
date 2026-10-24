"""Build readable and portable compressed single-cell notebooks from one source."""
from pathlib import Path
import ast,base64,hashlib,json,sys,textwrap,zlib

source_path=Path(sys.argv[1]);destination=Path(sys.argv[2]);destination.mkdir(parents=True,exist_ok=True)
source=source_path.read_text(encoding='utf-8');digest=hashlib.sha256(source.encode()).hexdigest()
tree=ast.parse(source)
assert not any(isinstance(node,ast.ImportFrom) and node.module=='__future__' for node in tree.body)
starts=[min([node.lineno]+[decorator.lineno for decorator in getattr(node,'decorator_list',[])])-1 for node in tree.body]
bounds=[0];begin=0
for index,node in enumerate(tree.body):
    if node.end_lineno-begin>=300 and index+1<len(tree.body):
        begin=starts[index+1];bounds.append(begin)
bounds.append(len(source.splitlines(keepends=True)))
payload=base64.b64encode(zlib.compress(source.encode(),9)).decode('ascii')
launcher='''# Portable single-cell application. Readable code is supplied in the companion notebook.
import ast as _holo_ast, base64 as _holo_b64, zlib as _holo_zlib
import hashlib as _holo_hashlib, linecache as _holo_linecache
_holo_expected_sha = '''+repr(digest)+'''
_holo_payload = (
'''+''.join('    '+repr(line)+'\n' for line in textwrap.wrap(payload,2000))+''')
_holo_source = _holo_zlib.decompress(_holo_b64.b64decode(_holo_payload)).decode('utf-8')
if _holo_hashlib.sha256(_holo_source.encode()).hexdigest() != _holo_expected_sha:
    raise RuntimeError('Notebook source verification failed.')
_holo_existing_root = globals().get('root')
_holo_reuse = False
if globals().get('_holo_active_sha') == _holo_expected_sha and _holo_existing_root is not None:
    try:
        _holo_reuse = bool(_holo_existing_root.winfo_exists())
    except Exception:
        pass
if _holo_reuse:
    _holo_existing_root.deiconify()
    _holo_existing_root.lift()
    print('GUI is already open; loaded data and measurements retained.', flush=True)
else:
    globals().pop('app', None)
    globals().pop('root', None)
    globals().pop('_holo_active_sha', None)
    _holo_filename = '<finalized_2d_analysis_readable.py>'
    _holo_lines = _holo_source.splitlines(keepends=True)
    _holo_linecache.cache[_holo_filename] = (len(_holo_source), None, _holo_lines, _holo_filename)
    _holo_bounds = '''+repr(bounds)+'''
    _holo_cached = globals().get('_holo_compiled_cache')
    if _holo_cached is not None and _holo_cached[0] == _holo_expected_sha:
        for _holo_code in _holo_cached[1]:
            exec(_holo_code, globals())
    else:
        _holo_codes = []
        for _holo_begin, _holo_end in zip(_holo_bounds, _holo_bounds[1:]):
            _holo_tree = _holo_ast.parse(''.join(_holo_lines[_holo_begin:_holo_end]), _holo_filename)
            if _holo_begin:
                _holo_ast.increment_lineno(_holo_tree, _holo_begin)
            _holo_code = compile(_holo_tree, _holo_filename, 'exec', dont_inherit=True)
            exec(_holo_code, globals())
            _holo_codes.append(_holo_code)
        _holo_compiled_cache = (_holo_expected_sha, tuple(_holo_codes))
    _holo_active_sha = _holo_expected_sha
'''
compile(launcher,'<compressed notebook>','exec')
for kind,text in [('readable',source),('compressed',launcher)]:
    book=dict(nbformat=4,nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python 3 (ipykernel)',language='python',name='python3'),
            language_info=dict(name='python',version='3.12'),
            finalized_runtime=dict(sha256=digest,format=kind,encoding='plain Python source' if kind=='readable' else 'zlib+base64')),
        cells=[dict(cell_type='code',id='finalized-'+kind,metadata={},execution_count=None,outputs=[],source=text.splitlines(keepends=True))])
    (destination/f'finalized_2d_analysis_{kind}.ipynb').write_text(json.dumps(book,indent=1,ensure_ascii=False),encoding='utf-8')
(destination/'finalized_2d_analysis_readable.py').write_text(source,encoding='utf-8')
assert zlib.decompress(base64.b64decode(payload)).decode()==source
(destination/'notebook_source_sha256.txt').write_text(digest+'  Application source shared by both notebooks\n',encoding='utf-8')
print(json.dumps(dict(source_sha256=digest,chunks=len(bounds)-1,files=[str(path) for path in destination.glob('finalized_2d_analysis_*')]),indent=2))
