"""Offline checks run on the source, portable executable, and single-file EXE."""
from __future__ import annotations

import json
import platform
from pathlib import Path
import struct
import sys
import tempfile
import time
import traceback


def run_self_test(report_path, *, gpu=False):
    result = dict(passed=False, frozen=bool(getattr(sys,'frozen',False)),
                  architecture_bits=struct.calcsize('P')*8, machine=platform.machine(),
                  python=platform.python_version(), checks=[])
    root = app = None
    try:
        from .vtk_compat import register_overrides
        register_overrides()
        import numpy as np
        import pyvista as pv
        import vtk
        import scipy.spatial
        import numba
        import llvmlite.binding
        from PIL import Image
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import cv2
        from . import core
        assert struct.calcsize('P')==8
        result['versions'] = dict(numpy=np.__version__,pyvista=pv.__version__,vtk=vtk.vtkVersion.GetVTKVersion(),numba=numba.__version__,opencv=cv2.__version__)
        result['checks'].append('x64 scientific and rendering libraries import')
        implicit = pv.ImageData(dimensions=(3,3,3)).GetPoints().GetData()
        assert hasattr(implicit, '__array__'), 'VTK implicit-array Python overrides were not loaded.'
        assert np.asarray(implicit).shape == (27,3)
        result['checks'].append('VTK implicit-array NumPy overrides in frozen import environment')
        with tempfile.TemporaryDirectory(prefix='mumax-exe-selftest-') as temp:
            folder=Path(temp);frames=folder/'fields';frames.mkdir();output=folder/'output';output.mkdir()
            z,y,x=np.indices((3,7,9))
            vectors=np.stack((np.ones_like(x),np.sin(y/3),np.linspace(-1,1,9)[x]),axis=-1).astype(np.float32)
            header='# OOMMF OVF 2.0\n# meshtype: rectangular\n# xnodes: 9\n# ynodes: 7\n# znodes: 3\n# valuedim: 3\n# meshunit: nm\n# xbase: 1\n# ybase: 1\n# zbase: .5\n# xstepsize: 2\n# ystepsize: 2\n# zstepsize: 1\n# Desc: Total simulation time: 2 ns\n'
            source=frames/'frame.ovf'
            source.write_text(header+'# Begin: Data Text\n'+'\n'.join(' '.join(map(str,v)) for v in vectors.reshape(-1,3))+'\n# End: Data Text\n',encoding='ascii')
            actual,mesh,timestamp,_=core.read_field_file(source)
            np.testing.assert_allclose(actual,vectors,atol=1e-7);assert timestamp==2.
            cache=core.Tab4VTKCache()
            try:
                converted,path=cache.convert_and_read(source)
                np.testing.assert_allclose(converted[0],vectors,atol=1e-7)
                assert path.is_file() and path.suffix=='.vtk'
            finally:cache.close()
            oom=frames/'field.omf';oom.write_bytes(source.read_bytes())
            assert len(core.field_files(frames))==2
            np.testing.assert_array_equal(core.parse_ovf_vectors(oom),vectors)
            result['checks'].append('OVF/OOMMF read, native VTK conversion/read, nm coordinates and cache cleanup')
            # Non-Cartesian sampling needs SciPy's frozen extension modules.
            distance,_=scipy.spatial.cKDTree(np.array([[0.,0.,0.],[1.,0.,0.]])).query([[.25,0.,0.]])
            np.testing.assert_allclose(distance,[.25])
            grid,_,_=core.magnetization_vtk_grid(vectors,mesh)
            assert grid.contour([0.],scalars='mz').n_points>0
            lines=grid.streamlines(vectors='m',n_points=8,max_length=5.)
            assert lines.n_points>0
            result['checks'].append('VTK CPU contours/streamlines and SciPy spatial extension')
            images=folder/'images';images.mkdir()
            for name in ('Cool to Warm','Black-Body Radiation'):
                image=core.scalar_slice_image(vectors[0,:,:,2],name)
                core.decorate_image(image,dict(colorbar=True),colors=name,component='mz').save(images/(name+'.png'))
            assert len(core.PARAVIEW_COLOR_PRESETS)==137
            _,movie,count=core.convert_images_to_video(images,output,8.,'360p')
            cap=cv2.VideoCapture(str(movie));assert cap.read()[0] and count==2;cap.release()
            result['checks'].append('ParaView presets, Pillow PNG, color bar and OpenCV MP4 encoder/decoder')
            root=core.tk.Tk();root.withdraw();app=core.ConverterGUI(root)
            callbacks=[];root.report_callback_exception=lambda *a:callbacks.append(str(a))
            assert len(app.stream_tab.master.tabs())==4
            for tab in (app.cross_tab,app.stream_tab):
                assert hasattr(tab,'layer_fill_control') and hasattr(tab,'arrow_transparency')
                assert hasattr(tab,'light_x_angle') and hasattr(tab,'arrow_reflection')
            assert len(app.color_box['values'])==len(core.STREAM_COLOR_MAPS)
            result['checks'].append('Tk and four complete tabs, ROI/layers, shared presets, lighting/reflection/transparency controls')
            for tab in (app.cross_tab,app.stream_tab):
                control=tab.layer_fill_control
                assert 'Glass' in control.finish_choice['values']
                control.sync_layers(mesh['z']);control.enabled.set(True)
                control.records[0]['checked']=True
                control.transparency.set('0');control.finish.set('Glass')
                control.finish_choice.event_generate('<<ComboboxSelected>>')
                record=control.snapshot()['layers'][0]
                assert record['finish']=='Glass' and record['transparency']==70.
                control.transparency.set('45')
                control.finish_choice.event_generate('<<ComboboxSelected>>')
                record=control.snapshot()['layers'][0]
                assert record['transparency']==45.
                planes=core.build_layer_fill_planes(vectors,mesh,mesh['z'],control.snapshot(),
                    'mz','blue,white,red',(-1,1))
                assert len(planes)==1 and np.isclose(planes[0]['opacity'],.55)
                assert planes[0]['z_bounds'][1]>planes[0]['z_bounds'][0]
                material=core.layer_finish_material(core.reflected_layer_record(record,core.REFLECTION_DEFAULTS),.3)
                assert np.isclose(material['specular'],.7) and material['specular_power']==120.
                actor=pv.Actor();core.apply_layer_finish(actor,record,.3)
                assert np.isclose(actor.prop.specular,.7) and actor.prop.GetSpecularColor()==(1.,1.,1.)
                override=core.reflected_layer_record(record,dict(core.REFLECTION_DEFAULTS,enabled=True,layer=20.,sharpness=35.))
                assert np.isclose(core.layer_finish_material(override,.3)['specular'],.14)
                assert core.layer_finish_material(override,.3)['specular_power']==35.
                control.enabled.set(False)
            result['checks'].append('Glass finish in both tabs, adjustable translucency, full layer thickness and material highlights')
            core.warm_mumax_renderer();deadline=time.monotonic()+90
            while core._MUMAX_WARMING and time.monotonic()<deadline:
                root.update();time.sleep(.025)
            assert core._MUMAX_FAST_KERNEL is not None,core._MUMAX_KERNEL_ERROR
            fig=Figure(figsize=(3,2),dpi=100);canvas=FigureCanvasAgg(fig)
            ax=fig.add_subplot(111,projection='mumax_full_3d');ax.set(xlim=(-1,1),ylim=(-1,1),zlim=(-1,1));canvas.draw()
            artist=core.LayerFilledGlyphArtist(ax,np.array([[0.,0.,0.]]),np.array([[0.,0.,1.]]),np.array([1.]),np.ones(3),
                np.array([[.7,.2,.1,1.]]),np.array([.1]),np.array([.2]),.35,.3,'Smooth (64)')
            artist.force_cpu=True;artist.reflection_settings=dict(core.REFLECTION_DEFAULTS,enabled=True,arrow=70.)
            ax.add_artist(artist);rgba=artist.raster(120,90,ax.get_proj(),ax.bbox.frozen())
            assert rgba[:,:,3].any() and artist.last_engine=='compiled'
            artist.reflection_settings['transparency']=50.
            assert artist.raster(120,90,ax.get_proj(),ax.bbox.frozen())[:,:,3].any()
            result['checks'].append('Numba/LLVM runtime compilation and CPU opaque/translucent 3D arrows')
            glass_record=dict(index=0,mode='Solid color',color='#80cfff',finish='Glass',transparency=70.)
            artist.fill_planes=[dict(index=0,z=0.,z_bounds=(-.2,.2),xy=(-.8,.8,-.8,.8),
                rgba=np.array([[[128,207,255,255]]],np.uint8),uniform=True,opacity=.3,
                record=core.reflected_layer_record(glass_record,core.REFLECTION_DEFAULTS))]
            rgba=artist.raster(120,90,ax.get_proj(),ax.bbox.frozen())
            assert np.count_nonzero((rgba[:,:,3]>0)&(rgba[:,:,3]<255))>50
            result['checks'].append('CPU glass layer rendering with translucent arrows')
            if gpu:
                artist.force_cpu=False
                rgba=artist.raster(120,90,ax.get_proj(),ax.bbox.frozen());assert rgba[:,:,3].any()
                result['gpu'] = dict(engine=artist.last_engine,device=getattr(artist,'gpu_device','CPU fallback'),error=getattr(artist,'gpu_error',None))
                core.close_mumax_gpu_scenes(fig)
                result['checks'].append('GPU render path or analytic CPU fallback')
            assert not callbacks,callbacks
        result['passed']=True
    except Exception:
        result['error']=traceback.format_exc()
    finally:
        if app is not None:
            try:app.close()
            except Exception:result['passed']=False;result['cleanup_error']=traceback.format_exc()
        elif root is not None:
            root.destroy()
        report_path=Path(report_path).resolve();report_path.parent.mkdir(parents=True,exist_ok=True)
        report_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1
