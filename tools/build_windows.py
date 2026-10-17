"""Build, execute smoke tests, and package the Windows x64 distributions."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
APP_NAME = 'MuMax3-Image-Tools'


def run(command, **kwargs):
    subprocess.run(command, check=True, cwd=PROJECT, **kwargs)


def check_executable(path):
    with path.open('rb') as stream:
        assert stream.read(2) == b'MZ', 'Output is not a Windows executable.'
        stream.seek(0x3c); offset = struct.unpack('<I', stream.read(4))[0]
        stream.seek(offset); assert stream.read(4) == b'PE\0\0'
        assert struct.unpack('<H', stream.read(2))[0] == 0x8664, 'Output is not AMD64/x64.'


def collect_notices(output):
    notices = output/'licenses'; notices.mkdir(exist_ok=True)
    for name in ('numpy','pillow','opencv-python','matplotlib','pyvista','vtk','scipy','numba','llvmlite','pyinstaller','pyinstaller-hooks-contrib'):
        dist = metadata.distribution(name); destination = notices/name; destination.mkdir(exist_ok=True)
        for item in dist.files or ():
            text = str(item).replace('\\','/')
            if ('dist-info/' in text and any(term in item.name.upper() for term in ('LICENSE','COPYING','NOTICE'))):
                path = Path(dist.locate_file(item))
                if path.is_file():
                    relative = text.split('.dist-info/', 1)[1]
                    target = destination/relative; target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(path, target)
        (destination/'metadata.txt').write_text(f'{dist.metadata["Name"]} {dist.version}\nLicense: {dist.metadata.get("License", "See package license files")}\nHome: {dist.metadata.get("Home-page", "")}\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('both','onedir','onefile'), default='both')
    parser.add_argument('--skip-tests', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'win32' or struct.calcsize('P') != 8:
        raise SystemExit('Use 64-bit Python on Windows to build this application.')
    release = PROJECT/'release'; release.mkdir(exist_ok=True)
    artifacts = []
    for mode in (('onedir','onefile') if args.mode == 'both' else (args.mode,)):
        env = os.environ.copy(); env['MUMAX_BUNDLE_MODE'] = mode
        run([sys.executable,'-m','PyInstaller','--noconfirm','--distpath',str(PROJECT/'dist'/mode),
             '--workpath',str(PROJECT/'build'/mode), str(PROJECT/'MuMax3.spec')], env=env)
        executable = PROJECT/'dist'/mode/((APP_NAME+'/'+APP_NAME+'.exe') if mode=='onedir' else (APP_NAME+'.exe'))
        check_executable(executable)
        if not args.skip_tests:
            report = release/f'self-test-{mode}.json'
            run([str(executable),'--self-test',str(report)], timeout=240)
            result = json.loads(report.read_text(encoding='utf-8'))
            if not result.get('passed'): raise RuntimeError(f'Frozen application failed: {result}')
        if mode == 'onedir':
            bundle = executable.parent
            shutil.copy2(PROJECT/'README.md', bundle/'README.md')
            shutil.copy2(PROJECT/'THIRD_PARTY_NOTICES.txt', bundle/'THIRD_PARTY_NOTICES.txt')
            collect_notices(bundle)
            archive = Path(shutil.make_archive(str(release/(APP_NAME+'-windows-x64-portable')),'zip',root_dir=bundle.parent,base_dir=bundle.name))
            artifacts.append(archive)
        else:
            target = release/(APP_NAME+'-windows-x64.exe'); shutil.copy2(executable,target);artifacts.append(target)
    shutil.copy2(PROJECT/'README.md', release/'README.md')
    shutil.copy2(PROJECT/'THIRD_PARTY_NOTICES.txt', release/'THIRD_PARTY_NOTICES.txt')
    collect_notices(release)
    checksums = release/'SHA256SUMS.txt'
    checksums.write_text(''.join(hashlib.file_digest(path.open('rb'),'sha256').hexdigest()+'  '+path.name+'\n' for path in artifacts),encoding='utf-8')
    print('Windows x64 artifacts:', *artifacts, sep='\n')


if __name__ == '__main__':main()
