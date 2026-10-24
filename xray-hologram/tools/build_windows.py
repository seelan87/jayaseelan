"""Build and test Windows x64 distributions; no installed Python is needed to run them."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,struct,subprocess,sys

project=Path(__file__).resolve().parents[1]
name='XRay-Hologram-Reconstruction'

def run(command,**kwargs):
    print('RUN',subprocess.list2cmdline([str(x) for x in command]),flush=True)
    subprocess.run([str(x) for x in command],cwd=project,check=True,**kwargs)

def check_exe(exe,report):
    # Keep host Python, CUDA, and external FFmpeg out of discovery. A separate
    # user cache verifies that bundled GPU runtimes also work on this computer.
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','CUDA_PATH','CUDA_HOME','IMAGEIO_FFMPEG_EXE'):
        env.pop(key,None)
    env['PATH']=os.pathsep.join((str(Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32'),
                                os.environ.get('SystemRoot',r'C:\Windows')))
    fresh=project/'test-results'/'clean-profile';fresh.mkdir(parents=True,exist_ok=True)
    env['USERPROFILE']=str(fresh)
    run([exe,'--self-test',report],env=env,timeout=240)
    data=json.loads(report.read_text())
    if not data.get('passed'):raise RuntimeError(str(data))
    with exe.open('rb') as stream:
        if stream.read(2)!=b'MZ':raise RuntimeError('Not a Windows executable')
        stream.seek(0x3c);offset=int.from_bytes(stream.read(4),'little');stream.seek(offset)
        if stream.read(4)!=b'PE\x00\x00' or int.from_bytes(stream.read(2),'little')!=0x8664:
            raise RuntimeError('Executable is not Windows x64')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('onedir','onefile','both'),default='both')
    args=parser.parse_args()
    if sys.platform!='win32' or struct.calcsize('P')!=8:raise RuntimeError('Use 64-bit Python on Windows')
    release=project/'release';release.mkdir(exist_ok=True)
    reports=project/'test-results';reports.mkdir(exist_ok=True)
    for mode in (('onedir','onefile') if args.mode=='both' else (args.mode,)):
        env=os.environ.copy();env['XRAY_BUNDLE_MODE']=mode
        run([sys.executable,'-m','PyInstaller','--noconfirm',
             '--workpath',project/'build'/mode,'--distpath',project/'dist'/mode,
             project/'XRayHologram.spec'],env=env)
        exe=project/'dist'/mode/(name+'.exe' if mode=='onefile' else name+'/'+name+'.exe')
        report=reports/('self-test-'+mode+'.json');check_exe(exe,report)
        shutil.copy2(report,release/report.name)
        if mode=='onefile':shutil.copy2(exe,release/(name+'-windows-x64.exe'))
        else:
            shutil.make_archive(str(release/(name+'-windows-x64-portable')),'zip',
                                root_dir=exe.parent.parent,base_dir=name)
    for filename in ('README.md','THIRD_PARTY_NOTICES.txt','source-provenance.json'):
        shutil.copy2(project/filename,release/filename)
    files=[p for p in release.iterdir() if p.suffix in ('.exe','.zip')]
    (release/'SHA256SUMS.txt').write_text(''.join(
        hashlib.file_digest(p.open('rb'),'sha256').hexdigest()+'  '+p.name+'\n' for p in files),encoding='utf-8')
    print('Verified downloads:',release,flush=True)

if __name__=='__main__':main()
