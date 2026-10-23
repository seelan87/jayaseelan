"""Regression checks for direct 2D analysis in source and frozen applications."""
from pathlib import Path
from unittest.mock import patch
import time
import numpy as np


def run_analysis_checks(core, app, root, check, output):
    yy, xx = np.indices((129, 129), dtype=float)
    plane = -8000. + 13.*xx - 4.*yy
    circle = 40000.*np.exp(-((xx-64.)**2+(yy-64.)**2)/(2*8.**2))
    ellipse = 40000.*np.exp(-((xx-64.)**2/(2*14.**2)+(yy-64.)**2/(2*3.5**2)))
    noise = np.random.default_rng(441).normal(0, 120., xx.shape)
    raw = np.stack([plane+circle, plane+ellipse, plane+noise, plane-circle])
    app.recons = raw.copy()
    app.real_recons = raw.copy()
    app.imag_recons = raw.copy()
    app.amplitude_recons = np.abs(raw)
    app.phase_recons = np.angle(raw+1j*raw)
    app.complex_recons = raw+1j*raw
    app._tab4_component_source_stacks = None
    app._tab4_component_source_signature = None
    app.supportmask2 = np.ones(raw.shape[1:], bool)
    app.supportmask = app.supportmask2.copy()
    app.real_fields_mT = np.arange(len(raw), dtype=float)
    app.current_values_A = np.zeros(len(raw))
    app.reconstruction_paths = []
    app.data_dir = Path(output)
    app.filter_var.set('None')
    app.brightness_var.set(0)
    app.contrast_var.set(1)
    # Production confirmation intentionally displays informational dialogs.
    with patch('tkinter.messagebox.showinfo', return_value=None):
        app.tab4_raw_intensity_check.invoke()
        app.tab4_component_var.set('Imaginary')
        core._tab4_confirm_component(app)
        app.roi_coords = dict(xmin=0, xmax=128, ymin=0, ymax=128)
        app.confirm_roi()
        app.pixel_cal_direct_nm_per_pixel_var.set('2')
        app._pixel_cal_apply_direct()
        app.roi_width_var.set('1024')
        app.roi_height_var.set('1024')
        app.apply_roi_resolution_from_gui()
    check(app.roi_stack.shape == (4, 1024, 1024), 'Selected imaginary ROI is processed at 1024 x 1024')
    app.filter_var.set('Gaussian')
    core._proc_reset(app)
    check(app.filter_var.get() == 'None', 'Tab 6 Reset restores Filter None after a manual filter selection')
    app.real_fields_mT = np.array([88.25, -7.125, 155.5, np.nan])
    check(core._tab9_roi4_fetch_from_tab6(app), 'Processed primary ROI reaches Tab 9')
    bounds = dict(xmin=int(40*1024/129), xmax=int(89*1024/129)-1,
                  ymin=int(40*1024/129), ymax=int(89*1024/129)-1)
    app._tab9_roi4_coords = bounds.copy()
    check(core._tab9_roi4_confirm(app), 'Confirmed processed feature ROI reaches Tab 10')
    check(core._tab11_fetch_primary(app), 'Same processed primary ROI reaches Tab 11')
    app._tab11_rois = [bounds.copy()]
    core._tab11_update_roi_selector(app, 0)

    def pump(seconds=.25):
        until = time.perf_counter()+seconds
        while time.perf_counter() < until:
            root.update()
            time.sleep(.005)

    def forbidden(*args, **kwargs):
        raise AssertionError('Browsing a calculated frame must never calculate or fit')

    for tab in (10, 11):
        app.nb.select(app.nb.tabs()[tab-1])
        root.update()
        getattr(app, f'tab{tab}_frame_var').set(0)
        getattr(app, f'tab{tab}_integrated_intensity_2d_check').invoke()
        getattr(app, f'tab{tab}_2d_polarity_var').set('Bright')
        getattr(app, f'tab{tab}_2d_minimum_outer_circularity_var').set(.70)
        getattr(app, f'tab{tab}_start_var').set(0)
        getattr(app, f'tab{tab}_end_var').set(2)
        getattr(app, f'tab{tab}_zero_if_undetected_var').set(False)
        measures = []
        original = core._ii2d_measure

        def counted(*args, **kwargs):
            result = original(*args, **kwargs)
            measures.append(result)
            return result

        with patch.object(core, '_ii2d_measure', counted):
            getattr(core, f'_tab{tab}_track_all')(app)
        root.update()
        check(len(measures) == 3, f'Tab {tab}: selected frames are calculated exactly once')
        data = core._ii2d_dataframe(app, tab).copy(deep=True)
        check(list(data.frame) == [0], f'Tab {tab}: circularity filter rejects elongated and noise frames')
        check(data.iloc[0].ii2d_integrated > 0, f'Tab {tab}: signed full-ROI intensity is retained')
        cache = getattr(app, f'_tab{tab}_2d_display_cache')
        check(set(cache) == {(0, 0), (1, 0), (2, 0)}, f'Tab {tab}: rejected frames also retain display records')
        state = getattr(app, f'_tab{tab}_2d_render_state')
        image = state['main_image']
        colorbar = state['contrast_image'].colorbar
        axes = tuple(getattr(app, f'tab{tab}_fig').axes)
        with patch.object(core, '_ii2d_compute', forbidden), \
             patch.object(core, '_ii2d_measure', forbidden), \
             patch.object(core, '_ii2d_run', forbidden):
            core._ii2d_slider_pressed(app, tab)
            for index in (1, 2, 0, 2, 1):
                getattr(core, f'_tab{tab}_frame_changed')(app, index)
                label = getattr(app, f'tab{tab}_frame_label_var').get()
                check(f'{index+1}/4' in label and f'{app.real_fields_mT[index]:+.6g}' in label,
                      f'Tab {tab}: frame {index} label and non-monotonic field update immediately')
            pump(.30)
            check(np.array_equal(image.get_array(), measures[0][1]['crop']),
                  f'Tab {tab}: dragging performs no figure update or analysis')
            core._ii2d_slider_released(app, tab)
            root.update()
            check(np.array_equal(image.get_array(), measures[1][1]['crop']),
                  f'Tab {tab}: release displays the exact stored selected-frame image')
            check(state['main_image'] is image and state['contrast_image'].colorbar is colorbar
                  and tuple(getattr(app, f'tab{tab}_fig').axes) == axes,
                  f'Tab {tab}: browsing reuses images, colorbars and axes')
            core._ii2d_frame_changed(app, tab, 3, immediate=True)
            root.update()
            check('not calculated' in getattr(app, f'tab{tab}_ii_status_var').get().lower(),
                  f'Tab {tab}: uncalculated frame does not trigger analysis')
            check(core._ii2d_dataframe(app, tab).equals(data),
                  f'Tab {tab}: slider browsing preserves measurement tables')
        check(len(getattr(app, f'_tab{tab}_2d_measurement_cache')) <= 2,
              f'Tab {tab}: full floating-map cache remains bounded')
        y2 = tuple(getattr(app, f'tab{tab}_ii_y2_combo').cget('values'))
        check('I.I. Avg. Line Fit' in y2 and '2D-I.I. (a.u. nm²)' in y2,
              f'Tab {tab}: existing Y2 menu contains 1D and 2D intensity')
        getattr(app, f'tab{tab}_frame_var').set(0)
        getattr(app, f'tab{tab}_integrated_intensity_2d_check').invoke()
        getattr(app, f'tab{tab}_gaussian_var').set(True)
        getattr(app, f'tab{tab}_dfit_var').set(False)
        getattr(app, f'tab{tab}_ref_mode_var').set('6')
        getattr(core, f'_tab{tab}_fit_current')(app)
        root.update()
        axis = getattr(app, f'tab{tab}_ax_fit')
        profile = next(line for line in axis.lines if line.get_label() == 'Average full profile')
        values = np.asarray(profile.get_ydata(), float)
        lo, hi = axis.get_ylim()
        check(axis.get_aspect() == 'auto' and lo <= values.min() and hi >= values.max()
              and hi-lo > 30000., f'Tab {tab}: 1D Gaussian profile scale is restored after 2D mode')
