"""Windows x64 entry point, with offline checks for packaged builds."""
from __future__ import annotations
import argparse,logging,multiprocessing,os,sys,traceback
from pathlib import Path

def configure_runtime():
    base=Path(os.environ.get('LOCALAPPDATA',str(Path.home())))/'XRayHologram'
    logs=base/'logs';logs.mkdir(parents=True,exist_ok=True)
    log=logs/'app.log'
    if sys.stdout is None:sys.stdout=log.open('a',encoding='utf-8',buffering=1)
    if sys.stderr is None:sys.stderr=log.open('a',encoding='utf-8',buffering=1)
    logging.basicConfig(filename=log,level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',encoding='utf-8')
    os.environ.setdefault('MPLBACKEND','TkAgg')
    mpl=base/'matplotlib';mpl.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(mpl))
    os.environ.setdefault('OMP_NUM_THREADS',str(max(1,min(4,(os.cpu_count() or 2)//2))))
    if not getattr(sys,'frozen',False):
        sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
    return log

def main(argv=None):
    log=configure_runtime()
    parser=argparse.ArgumentParser(description='X-Ray Hologram Reconstruction for Windows x64')
    parser.add_argument('--self-test',metavar='REPORT_JSON')
    args=parser.parse_args(argv)
    try:
        if args.self_test:
            from xray_app.self_test import run_self_test
            return run_self_test(Path(args.self_test))
        from xray_app import core
        def callback_error(kind,value,tb):
            logging.error('GUI callback failed',exc_info=(kind,value,tb))
            core.messagebox.showerror('X-Ray Hologram Reconstruction',f'{value}\n\nDetails: {log}',parent=core.root)
        core.root.report_callback_exception=callback_error
        core.root.mainloop()
        return 0
    except Exception:
        logging.exception('Application failed')
        traceback.print_exc()
        if sys.platform=='win32':
            import ctypes
            ctypes.windll.user32.MessageBoxW(None,f'Could not open X-Ray Hologram Reconstruction.\n\nDetails: {log}',
                'X-Ray Hologram Reconstruction',0x10)
        return 1

if __name__=='__main__':
    multiprocessing.freeze_support()
    raise SystemExit(main())
