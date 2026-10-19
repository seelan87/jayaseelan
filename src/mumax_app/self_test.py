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


def check_stream_lighting(core,view,pv,np,report_path):
    """Exercise the actual Tab 4 rebuild and cached-display paths on the GPU."""
    paint=view.paint;view.paint=lambda:None
    sphere=pv.Sphere(radius=.65,theta_resolution=32,phi_resolution=24);sphere['mz']=np.zeros(sphere.n_points)
    angle=np.linspace(0,4*np.pi,160)
    points=np.column_stack((.85*np.cos(angle),.85*np.sin(angle),np.linspace(-.8,.8,len(angle))))
    lines=pv.PolyData(points,lines=np.r_[len(points),np.arange(len(points))]);lines['mz']=np.zeros(len(points))
    box=pv.ImageData(dimensions=(3,3,3),origin=(-1,-1,-1),spacing=(1,1,1))
    view.plotter=pv.Plotter(off_screen=True,window_size=(600,420),shape=(1,2),col_weights=[.88,.12],border=False)
    view.style=view.filter_settings()[1]
    view.style.update(scales=False,bar=False,outline=False,seeds=False,opacity=1.,stream_opacity=1.,radius=.045,
        line_width=5.,ambient_light=.15,colors='Blue → White → Red',background='#202020',
        light=dict(core.LIGHT_DEFAULTS,height_m=1e-9,intensity=1.),
        reflection=dict(core.REFLECTION_DEFAULTS,enabled=True,layer=80.,arrow=60.,sharpness=30.))
    view.result=dict(grid=box,iso=sphere,streams=lines,seeds=pv.PolyData(),scalar_range=(-1.,1.),display_bounds=box.bounds,section=None)
    def capture():
        view.display_scene();view.plotter.render()
        assert view.plotter.renderers[0].GetLights().GetNumberOfItems()==2
        return np.asarray(view.plotter.screenshot(return_img=True)).copy()
    def differs(a,b):return np.count_nonzero(np.max(np.abs(a.astype(float)-b.astype(float)),axis=2)>5)>50
    try:
        first=capture();actors={k:a[0] for k,a in view.actor_cache.items()}
        view.style['light'].update(x_angle=65.,y_angle=110.);angled=capture()
        assert differs(first,angled)
        assert all(view.actor_cache[k][0] is actor for k,actor in actors.items())
        view.style['light']['intensity']=0.;unlit=capture()
        view.style['ambient_light']=.85;ambient=capture();assert differs(unlit,ambient)
        view.style['ambient_light']=.15;view.style['light']['intensity']=1.
        view.style['reflection']['enabled']=False;matte=capture()
        view.style['reflection'].update(enabled=True,layer=100.,arrow=100.,sharpness=8.);gloss=capture()
        assert differs(matte,gloss)
        for kind in ('iso','streams'):
            prop=view.actor_cache[kind][0].prop
            assert prop.lighting and np.isclose(prop.specular,.85) and prop.specular_power==8.
        for representation in ('Tubes','Lines'):
            view.style['stream_style']=representation;capture()
            view.actor_cache['iso'][0].visibility=False
            view.style['light'].update(x_angle=0.,y_angle=0.);before=capture()
            view.style['light'].update(x_angle=85.,y_angle=125.);after=capture();assert differs(before,after),representation
            if representation=='Lines':assert view.actor_cache['streams'][0].prop.GetRenderLinesAsTubes()
        angle=np.linspace(0,2*np.pi,97);points=np.column_stack((.7*np.cos(angle),.7*np.sin(angle),np.zeros(len(angle))))
        contour=pv.PolyData(points,lines=np.r_[len(points),np.arange(len(points))]);contour['mz']=np.zeros(len(points))
        view.result=dict(view.result,iso=contour,streams=pv.PolyData(),section=dict(view='XY',mode='Plane',position=0.),slice=pv.Plane())
        view.style['section_visual']=dict(show_slice=False,slice_opacity=.35)
        view.style['light'].update(x_angle=0.,y_angle=0.);before=capture()
        prop=view.actor_cache['iso'][0].prop;assert prop.lighting and prop.GetRenderLinesAsTubes()
        view.style['light'].update(x_angle=85.,y_angle=125.);after=capture();assert differs(before,after),'slice contours'
        folder=Path(report_path).resolve().parent;folder.mkdir(parents=True,exist_ok=True)
        from PIL import Image
        Image.fromarray(first).save(folder/'tab4-lighting-before.png');Image.fromarray(angled).save(folder/'tab4-lighting-after.png')
    finally:
        view.plotter.close();view.plotter=None;view.paint=paint;view.scene_valid=False
        view.actor_cache={};view.outline_actor=view.bounds_actor=None


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
            assert len(core.PARAVIEW_COLOR_PRESETS)==139
            for name,endpoints in (('Red → White → Blue',[[1.,0.,0.],[1.,1.,1.],[0.,0.,1.]]),
                    ('Blue → White → Red',[[0.,0.,1.],[1.,1.,1.],[1.,0.,0.]])):
                np.testing.assert_allclose(core.stream_color_map(name)(np.array([0.,.5,1.]))[:,:3],endpoints,atol=1e-6)
                colored=core.render_field_image(source,colors=name,annotations=dict(colorbar=True))
                assert colored.mode=='RGB' and colored.width>vectors.shape[2]
            result['checks'].append('Shared exact Red/White/Blue and Blue/White/Red presets with matching field colors and color bars')
            _,movie,count=core.convert_images_to_video(images,output,8.,'360p')
            cap=cv2.VideoCapture(str(movie));assert cap.read()[0] and count==2;cap.release()
            result['checks'].append('ParaView presets, Pillow PNG, color bar and OpenCV MP4 encoder/decoder')
            root=core.tk.Tk();root.withdraw();app=core.ConverterGUI(root)
            callbacks=[];root.report_callback_exception=lambda *a:callbacks.append(str(a))
            assert len(app.stream_tab.master.tabs())==4
            for tab in (app.cross_tab,app.stream_tab):
                assert hasattr(tab,'layer_fill_control') and hasattr(tab,'arrow_transparency')
                assert hasattr(tab,'light_x_angle') and hasattr(tab,'arrow_reflection')
            def descendants(widget):
                for child in widget.winfo_children():
                    yield child
                    yield from descendants(child)
            for owner,parent,open_picker in ((app,app.ovf_tab,app.choose_tab1_color_preset),
                    (app.cross_tab,app.cross_tab.controls,app.cross_tab.choose_color_preset),
                    (app.stream_tab,app.stream_tab.controls,app.stream_tab.choose_color_preset)):
                assert not any(isinstance(w,core.ttk.Combobox) and str(w.cget('textvariable'))==str(owner.colors)
                    for w in descendants(parent))
                open_picker()
                picker=getattr(owner,'preset_picker',None) if owner is app.stream_tab else owner.color_preset_picker
                assert set(picker.names)==set(core.PARAVIEW_COLOR_PRESETS)
                picker.tree.selection_set('Cool to Warm');picker.apply_selected()
                assert owner.colors.get()=='Cool to Warm'
            stream=app.stream_tab
            import tkinter.colorchooser as colorchooser
            original_picker=colorchooser.askcolor
            try:
                colorchooser.askcolor=lambda **kwargs:((0,0,0),'#000000')
                stream.background_button.invoke();assert stream.background_color.get()=='#000000'
                colorchooser.askcolor=lambda **kwargs:((255,128,0),'#ff8000')
                stream.outline_button.invoke();assert stream.outline_color.get()=='#ff8000'
                colorchooser.askcolor=lambda **kwargs:(None,None)
                stream.outline_button.invoke();assert stream.outline_color.get()=='#ff8000'
            finally:colorchooser.askcolor=original_picker
            stream.outline_checkbox.invoke();assert not stream.show_outline.get()
            stream.outline_checkbox.invoke();assert stream.show_outline.get()
            _,display_style=stream.filter_settings()
            assert display_style['background']=='#000000' and display_style['outline_color']=='#ff8000'
            assert display_style['outline']
            calls=[];bar=vtk.vtkScalarBarActor();bounds_actor=vtk.vtkCubeAxesActor();outline=pv.Actor()
            display=core.SimpleNamespace(style=display_style,bounds_actor=bounds_actor,outline_actor=outline,
                plotter=core.SimpleNamespace(set_background=lambda *args,**kwargs:calls.append((args,kwargs)),scalar_bars={'m':bar}))
            core.apply_stream_display_colors(display)
            assert calls[-1][0][0]=='#000000' and bar.GetLabelTextProperty().GetColor()==(1.,1.,1.)
            assert bounds_actor.GetTitleTextProperty(0).GetColor()==(1.,1.,1.)
            np.testing.assert_allclose(outline.prop.GetColor(),(1.,128/255,0.))
            display.style=dict(display_style,background='#ffffff',outline_color='#00ffff')
            core.apply_stream_display_colors(display)
            assert bar.GetLabelTextProperty().GetColor()==(0.,0.,0.)
            np.testing.assert_allclose(outline.prop.GetColor(),(0.,1.,1.))
            title=core.draw_figure_header(Image.new('RGB',(240,100)),['mz','t = 2 ns'],12,
                background='#000000',foreground='white')
            assert title.getpixel((0,0))==(0,0,0) and np.asarray(title).max()==255
            stream.background_color.set('#ffffff');stream.outline_color.set('#303030')
            result['checks'].append('Tab 4 background/outline controls, display/export style snapshots and dark-background label contrast')
            renderer=vtk.vtkRenderer()
            lighting=core.SimpleNamespace(plotter=core.SimpleNamespace(renderers=[renderer]),mesh=mesh,
                result=dict(display_bounds=(-1.,1.,-1.,1.,-1.,1.)),layer_fill_actors={},
                style=dict(light=dict(core.LIGHT_DEFAULTS),ambient_light=.2,
                    reflection=dict(core.REFLECTION_DEFAULTS,enabled=True,layer=80.,arrow=60.,sharpness=77.)),
                actor_cache={k:(pv.Actor(),False) for k in ('iso','streams','arrows')})
            core.apply_stream_light_controls(lighting);assert renderer.GetLights().GetNumberOfItems()==2
            old_position=lighting.scene_lights[0].GetPosition();renderer.RemoveAllLights()
            lighting.style['light'].update(x_angle=65.,y_angle=110.,intensity=1.5)
            core.apply_stream_light_controls(lighting)
            assert renderer.GetLights().GetNumberOfItems()==2 and lighting.scene_lights[0].GetPosition()!=old_position
            assert np.isclose(lighting.scene_lights[0].GetIntensity(),.85*1.5)
            for actor,_ in lighting.actor_cache.values():actor.prop.lighting=False
            core.apply_stream_reflection(lighting)
            for kind,(actor,_) in lighting.actor_cache.items():
                assert actor.prop.lighting and np.isclose(actor.prop.ambient,.2) and np.isclose(actor.prop.diffuse,.8)
                assert np.isclose(actor.prop.specular,.64 if kind=='iso' else .48) and actor.prop.specular_power==77.
            lighting.style['reflection']['enabled']=False;core.apply_stream_reflection(lighting)
            assert all(a.prop.specular==0. and a.prop.lighting for a,_ in lighting.actor_cache.values())
            result['checks'].append('Tab 4 restores scene lights after clearing and applies ambient/reflection to contours, streams and heads')
            assert stream.view_3d_group not in stream.section_groups and stream.view_3d_group.winfo_manager()=='pack'
            assert not any(isinstance(w,core.ttk.Combobox) and str(w.cget('textvariable'))==str(stream.camera_mode)
                for group in stream.section_groups for w in descendants(group))
            calls=[]
            class CameraProbe:
                def __getattr__(self,name):return lambda:calls.append(name)
            stream.plotter=CameraProbe()
            try:
                for section in (False,True):
                    stream.cross_section_enabled.set(section)
                    for view in ('XY','YZ','XZ','3D'):
                        stream.volume_view.set(view);stream.volume_view_changed();calls.clear();stream.camera_default()
                        assert calls[0]=={'XY':'view_xy','YZ':'view_yz','XZ':'view_xz','3D':'view_isometric'}[view]
                        assert stream.camera_mode.get()=='3D'
            finally:
                stream.plotter=None;stream.cross_section_enabled.set(False);stream.camera_mode.set('3D')
            result['checks'].append('ParaView-only preset pickers in Tabs 1/3/4 and independent XY/YZ/XZ/3D camera controls')
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
                control.edges.set(True);control.commit();record=control.snapshot()['layers'][0]
                assert record['edges']
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
            slab=dict(index=0,z=.5,z_bounds=(0.,1.),xy=(0.,1.,0.,1.),rgba=np.array([[[100,150,200,255]]],np.uint8),
                uniform=True,opacity=.3,record=dict(mode='Solid color',color='#6496c8',finish='Glass',edges=True))
            assert len(core.layer_fill_edge_segments(slab))==12
            assert core.vtk_layer_fill_edges(slab).n_lines==12
            assert core.vtk_layer_fill_edges(slab).n_verts==0
            merged=core.merge_layer_fill_planes([slab,dict(slab,index=1,z=1.5,z_bounds=(1.,2.))])
            assert len(merged)==1 and len(core.layer_fill_edge_segments(merged[0]))==12
            assert not np.any(np.all(np.isclose(core.layer_fill_edge_segments(merged[0])[:,:,2],1.),axis=1))
            perforated=np.full((3,3,4),255,np.uint8);perforated[1,1,3]=0
            hole=core.layer_fill_edge_segments(dict(slab,rgba=perforated,uniform=False))
            assert any(np.allclose(edge[:,0],1/3) and np.allclose(edge[:,1],1/3) for edge in hole)
            result['checks'].append('Layer outer edges preserve holes and remove internal boundaries of merged layers')
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
            artist.fill_planes[0]['record']['edges']=True
            edged=artist.raster(120,90,ax.get_proj(),ax.bbox.frozen())
            assert np.count_nonzero(np.all(edged[:,:,:3]<30,axis=2)&(edged[:,:,3]>0))>20
            result['checks'].append('CPU glass layer rendering with translucent arrows')
            if gpu:
                artist.force_cpu=False
                rgba=artist.raster(120,90,ax.get_proj(),ax.bbox.frozen());assert rgba[:,:,3].any()
                assert np.count_nonzero(np.all(rgba[:,:,:3]<30,axis=2)&(rgba[:,:,3]>0))>20
                result['gpu'] = dict(engine=artist.last_engine,device=getattr(artist,'gpu_device','CPU fallback'),error=getattr(artist,'gpu_error',None))
                core.close_mumax_gpu_scenes(fig)
                result['checks'].append('GPU render path or analytic CPU fallback')
                stream=app.stream_tab;paint=stream.paint
                stream.vectors=vectors;stream.mesh=mesh;stream.time_ns=2.
                stream.style=stream.filter_settings()[1]
                stream.result=dict(grid=grid,iso=grid.contour([0.],scalars='mz'),streams=pv.PolyData(),seeds=pv.PolyData(),
                    scalar_range=(-1.,1.),display_bounds=grid.bounds,section=None)
                stream.plotter=pv.Plotter(off_screen=True,window_size=(700,500),shape=(1,2),col_weights=[.88,.12],border=False)
                stream.paint=lambda:None
                try:
                    stream.style.update(background='#000000',outline_color='#ff8000',colors='Blue → White → Red')
                    stream.display_scene();stream.plotter.render()
                    dark=stream.overlay(Image.fromarray(stream.plotter.screenshot(return_img=True)))
                    assert dark.getpixel((0,0))==(0,0,0)
                    np.testing.assert_allclose(stream.outline_actor.prop.GetColor(),(1.,128/255,0.))
                    iso_actor=stream.actor_cache['iso'][0]
                    stream.style.update(background='#ffffff',outline_color='#00ffff')
                    stream.display_scene();stream.plotter.render()
                    light=stream.overlay(Image.fromarray(stream.plotter.screenshot(return_img=True)))
                    assert light.getpixel((0,0))==(255,255,255)
                    assert stream.actor_cache['iso'][0] is iso_actor
                    np.testing.assert_allclose(stream.outline_actor.prop.GetColor(),(0.,1.,1.))
                    stream.style['outline']=False;stream.display_scene()
                    assert stream.outline_actor is None
                    previews=Path(report_path).resolve().parent;previews.mkdir(parents=True,exist_ok=True)
                    dark.save(previews/'tab4-dark-background.png');light.save(previews/'tab4-light-background.png')
                    result['checks'].append('Tab 4 GPU render/export images, cached background/outline updates and outline visibility')
                finally:
                    stream.plotter.close();stream.plotter=None;stream.paint=paint;stream.scene_valid=False
                    stream.actor_cache={};stream.outline_actor=stream.bounds_actor=None
                check_stream_lighting(core,app.stream_tab,pv,np,report_path)
                result['checks'].append('GPU lighting changes for cached surfaces, streamline tubes/lines and slice contours; intensity, ambient and reflections')
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
