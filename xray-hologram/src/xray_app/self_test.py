"""Offline checks executed against source and both frozen distributions."""
from __future__ import annotations
from pathlib import Path
import json,platform,struct,tempfile,time,traceback,sys,os

def run_self_test(report_path):
    report_path=Path(report_path).resolve()
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report={'passed':False,'checks':[],'python':sys.version,'platform':platform.platform(),
            'frozen':bool(getattr(sys,'frozen',False))}
    core=None
    def check(condition,name):
        if not condition:raise AssertionError(name)
        report['checks'].append(name);print('PASS',name,flush=True)
    try:
        check(struct.calcsize('P')==8,'64-bit process')
        import numpy as np
        import pandas as pd
        import h5py
        import imageio.v2 as imageio
        import imageio_ffmpeg
        from PIL import Image
        from tkinter import messagebox
        dialogs=[]
        originals={name:getattr(messagebox,name) for name in ('showerror','showwarning','showinfo')}
        for name in originals:
            setattr(messagebox,name,lambda *a,_name=name,**k:dialogs.append((_name,str(a))))
        started=time.perf_counter()
        from xray_app import core
        root,app=core.root,core.app
        errors=[]
        root.report_callback_exception=lambda *a:errors.append(''.join(traceback.format_exception(*a)))
        root.update()
        report['gui_ready_s']=time.perf_counter()-started
        check(root.winfo_exists() and root.winfo_ismapped(),'GUI opens without Python/Jupyter integration')
        check(len(app.nb.tabs())==15,'All 15 tabs are built before ready')
        check(not core._startup_loading_window.winfo_exists(),'Loading screen is closed when all tabs are ready')
        check(not core._compute_gpu_checked,'CUDA stays unloaded during GUI startup')
        check(app.tab10_fix_center_button.cget('text')=='FIX CENTER','Tab 10 Fix Center control exists')
        check(app.magnet_fix_button.cget('text')=='Fix','Calibration Fix control exists')
        check(float(app.tab10_slider.cget('from'))==0,'Frame sliders use zero-based indexing')
        for tab in app.nb.tabs():
            app.nb.select(tab);root.update()
        for tab,notebook in ((app.tab_single_roi10,app.tab10_plot_views),
                             (app.tab_multi_roi11,app.tab11_plot_views)):
            app.nb.select(tab);root.update()
            for page in notebook.tabs():notebook.select(page);root.update()
        canvases=[c for c in core._figure_navigation_canvases if c.get_tk_widget().winfo_exists()]
        check(len(canvases)>=15,'Embedded figures exist across analysis tabs')
        opened=0
        for canvas in canvases:
            widget=canvas.get_tk_widget();ancestors=[]
            while widget.master is not None:
                parent=widget.master
                if isinstance(parent,core.ttk.Notebook):ancestors.append((parent,widget))
                widget=parent
            for notebook,page in reversed(ancestors):notebook.select(page)
            root.update()
            if canvas.get_tk_widget().winfo_ismapped():
                check(canvas.toolbar is not None,'Opened figure has navigation controls')
                opened+=1
        check(opened>=15,'All active figure pages open with toolbars')
        tick=[];root.after(30,lambda:tick.append(True));end=time.monotonic()+2
        while not tick and time.monotonic()<end:root.update();time.sleep(.01)
        check(tick==[True],'Tk callbacks execute normally')
        with tempfile.TemporaryDirectory(prefix='xray-self-test-') as tmp:
            tmp=Path(tmp)
            array=(np.arange(80).reshape(8,10)+1j).astype('complex64')
            with h5py.File(tmp/'data.h5','w') as h5:h5['reconstruction']=array
            with h5py.File(tmp/'data.h5','r') as h5:
                check(np.array_equal(h5['reconstruction'][()],array),'HDF5 complex data roundtrip')
            pd.DataFrame({'frame':[0,1],'field_mT':[-1.,1.]}).to_csv(tmp/'table.csv',index=False)
            check(len(pd.read_csv(tmp/'table.csv'))==2,'CSV table export and readback')
            calibration=tmp/'calibration.json'
            calibration.write_text(json.dumps({'current':[-4,0,4,0,-4],'field':[-42,-2,38,2,-38]}))
            app.field_calib_file=calibration;app.calib_var.set(str(calibration))
            check(app.load_calibration(),'Calibration JSON loads')
            check(app._magnet_selected_label=='Branch 2: +4 to -4 A','Calibration defaults to +4 A to -4 A')
            expected=app.field_interpolator([-1.,0.,1.])
            app.magnet_fix_button.invoke();calibration.unlink();app.new_upload()
            check(np.allclose(app.field_interpolator([-1.,0.,1.]),expected),'Fixed calibration survives New Upload and file removal')
            app.magnet_fix_button.invoke();app.new_upload()
            yy,xx=np.indices((81,81),dtype=float)
            crop=.15+.7*np.exp(-.5*(((xx-40)/7)**2+((yy-42)/9)**2))
            app._tab10_roi4_stack=crop[None].copy();app._tab10_roi4_scale=(2.,3.)
            app._tab10_roi4_coords=dict(xmin=0,xmax=80,ymin=0,ymax=80)
            app._tab11_primary_stack=crop[None].copy();app._tab11_primary_ready=True
            app._tab11_scale_nm=(2.,3.);app._tab11_rois=[dict(xmin=0,xmax=80,ymin=0,ymax=80)]
            app._tab11_center_fixed=True
            center=core._tab10_center(app,crop)
            app._tab10_center_cache=dict(center);app._tab11_centers={0:dict(center)}
            for tab in (10,11):getattr(app,f'tab{tab}_zero_if_undetected_var').set(False)
            for mode in ('gaussian','gradient','threshold'):
                for tab in (10,11):
                    getattr(app,f'tab{tab}_ref_mode_var').set('6')
                    getattr(app,f'tab{tab}_dfit_var').set(mode!='gaussian')
                    getattr(app,f'tab{tab}_gaussian_var').set(mode=='gaussian')
                    getattr(app,f'tab{tab}_dfit_sg_level_var').set(3)
                    getattr(app,f'tab{tab}_dfit_threshold_mode_var').set(mode=='threshold')
                    getattr(app,f'tab{tab}_dfit_threshold_var').set(.5)
                first=core._tab10_fit_roi4(app,0,detection=False)
                second=core._tab11_fit_roi(app,0,0)
                for key in ('t_nm','average_profile','fit_profile','fwhm','diameter_nm'):
                    check(np.allclose(first[key],second[key],equal_nan=True),f'Tab 10/11 {mode}: {key} agrees')
                core._tab10_draw_fit_result(app,first)
                core._tab11_draw_current_fit_views(app,second)
            app.tab10_fig.savefig(tmp/'figure.png',dpi=80)
            with Image.open(tmp/'figure.png') as im:
                check(im.width>100 and im.height>100,'Scientific figure PNG export')
                im.verify()
            ffmpeg=Path(imageio_ffmpeg.get_ffmpeg_exe())
            check(ffmpeg.is_file(),'FFmpeg is bundled and discoverable')
            with imageio.get_writer(tmp/'movie.mp4',fps=3,format='ffmpeg',macro_block_size=None) as writer:
                for value in (0,100,200):writer.append_data(np.full((32,48,3),value,dtype='uint8'))
            with imageio.get_reader(tmp/'movie.mp4',format='ffmpeg') as reader:
                check(sum(1 for _ in reader)==3,'MP4 encoding and decoding work offline')
            signal=np.random.default_rng(8).normal(size=(1024,1024)).astype('float32')
            transformed=core._compute_fft2(signal)
            check(np.allclose(transformed,np.fft.fft2(signal),rtol=3e-5,atol=.003),'Large FFT matches NumPy')
            check(np.allclose(core._compute_fft2(transformed,inverse=True).real,signal,rtol=2e-5,atol=3e-6),'FFT inverse roundtrip')
            report['fft_backend']=core._compute_last_backend
            report['gpu_available']=core._compute_gpu is not None
            report['gpu_unavailable_reason']=core._compute_gpu_error
            # Import must succeed even on a runner with no NVIDIA hardware.
            # A missing frozen module is a packaging error, not CPU fallback.
            import cupy as cp
            check(cp.__version__=='13.6.0','Bundled CuPy modules import successfully')
            if core._compute_gpu is not None:
                gpu_fft=cp.asnumpy(cp.fft.fft2(cp.asarray(signal)))
                check(np.allclose(gpu_fft,np.fft.fft2(signal),rtol=3e-5,atol=.003),'GPU FFT executes using bundled runtime libraries')
            if os.environ.get('XRAY_REQUIRE_GPU')=='1':
                check(core._compute_gpu is not None,'NVIDIA GPU backend available in packaged app: '+core._compute_gpu_error)
        check(not errors,'No Tk callback failures')
        check(not dialogs,'No unexpected error or warning dialogs')
        report['passed']=True
        return 0
    except Exception:
        report['error']=traceback.format_exc();print(report['error'],flush=True)
        return 1
    finally:
        if core is not None:
            try:
                for job in core.root.tk.call('after','info'):core.root.after_cancel(job)
                core.root.destroy()
            except Exception:pass
        report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
