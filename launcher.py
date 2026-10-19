"""Windows application entry point; safe to import and safe in a frozen process."""
from __future__ import annotations

import argparse
import io
import logging
import multiprocessing
import os
from pathlib import Path
import sys
import traceback


def configure_runtime():
    base = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'MuMax3ImageTools'
    base.mkdir(parents=True, exist_ok=True)
    logs = base / 'logs'; logs.mkdir(exist_ok=True)
    log = logs / 'app.log'
    # A --windowed executable has no stdout/stderr; native/Python libraries still
    # expect writable streams. Keep diagnostics outside the executable directory.
    if sys.stdout is None:
        sys.stdout = log.open('a', encoding='utf-8', buffering=1)
    if sys.stderr is None:
        sys.stderr = log.open('a', encoding='utf-8', buffering=1)
    logging.basicConfig(filename=log, level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s', encoding='utf-8')
    os.environ.setdefault('MPLBACKEND', 'TkAgg')
    mpl = base / 'matplotlib'; mpl.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(mpl))
    # Keep CPU parallelism bounded when numerical libraries and VTK share work.
    os.environ.setdefault('OMP_NUM_THREADS', str(max(1, min(8, (os.cpu_count() or 2)-1))))
    application_dir = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
    os.environ['PATH'] = os.pathsep.join((str(application_dir), str(application_dir/'bin'), os.environ.get('PATH', '')))
    if not getattr(sys, 'frozen', False):
        sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
    return log


def main(argv=None):
    log = configure_runtime()
    parser = argparse.ArgumentParser(description='MuMax3 Image Tools for Windows x64')
    parser.add_argument('--self-test', metavar='REPORT_JSON', help='Run offline dependency/I/O/UI/render/export checks and exit.')
    parser.add_argument('--render-test', action='store_true', help='Also exercise GPU rendering during the self-test.')
    args = parser.parse_args(argv)
    if args.self_test:
        from mumax_app.self_test import run_self_test
        return run_self_test(Path(args.self_test), gpu=args.render_test)
    try:
        from mumax_app.vtk_compat import register_overrides
        register_overrides()
        from mumax_app.core import ConverterGUI, tk, messagebox
        root = tk.Tk()
        def report_callback_exception(kind, value, tb):
            logging.error('GUI callback failed', exc_info=(kind, value, tb))
            messagebox.showerror('MuMax3 Image Tools', f'{value}\n\nDetails: {log}', parent=root)
        root.report_callback_exception = report_callback_exception
        ConverterGUI(root)
        root.mainloop()
        return 0
    except Exception:
        logging.exception('Application startup failed')
        # Use a native box even if Tk itself failed to import.
        if sys.platform == 'win32':
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, f'Could not start MuMax3 Image Tools.\n\nDetails: {log}', 'MuMax3 Image Tools', 0x10)
        else:
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    multiprocessing.freeze_support()
    raise SystemExit(main())
