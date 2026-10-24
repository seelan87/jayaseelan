"""Stored-measurement export regression checks for source and frozen builds."""
from pathlib import Path
from unittest.mock import patch
import json
import numpy as np
import pandas as pd


def run_export_checks(core, app, root, check, output):
    output=Path(output)/'measurement-exports';output.mkdir(parents=True,exist_ok=True)
    # A signed, non-Gaussian feature on a sloping background has an exact 2D integral.
    yy,xx=np.indices((81,81),dtype=float)
    plane=-4000+3*xx-5*yy
    square=(abs(xx-40)<=5)&(abs(yy-40)<=5)
    for sign in (1,-1):
        image=plane+sign*12000*square
        result,maps=core._ii2d_measure(image,dict(xmin=20,xmax=60,ymin=20,ymax=60),
            (2.,3.),background='Surrounding plane',fraction=.3,polarity='Bright' if sign>0 else 'Dark')
        check(np.isclose(result['ii2d_integrated'],sign*12000*121*6,rtol=1e-10),
              f'2D integral retains sign and independent X/Y pixel area ({sign:+d})')
        table=core._ii2d_export_table(pd.DataFrame([dict(result,frame=2,roi_index=1,field_mT=-7.125)]))
        check(table.Frame.iloc[0]==3 and table.Frame_Index.iloc[0]==2 and
              table.II_2D_au_nm2.iloc[0]==result['ii2d_integrated'], '2D table preserves frame identity and exact integral')
    # Every measured angle and particle must be retained across frames, not just the slider frame.
    app.pd_results={0:(1,-10.,120.123456789,.5,.04,25.,1,9876.54321,3.),
                    2:(3,10.,110.987654321,.7,.04,25.,1,-4567.12345,4.)}
    app.pd_frame_detections={i:[dict(x=20.,y=21.,diameter_nm=row[2],core_intensity=row[7],accepted=True)]
                             for i,row in app.pd_results.items()}
    app.pd_dfit_results={i:[dict(particle_no=1,avg_dia_nm=row[2],sd_nm=row[3],core_intensity=row[7],
        measurements=[dict(angle_deg=angle,radius_nm=60.+angle/100,diameter_nm=120.+angle/50,valid=True)
                      for angle in (0.,90.,180.,270.)])] for i,row in app.pd_results.items()}
    app._tab8_roi3_coords=dict(xmin=0,xmax=80,ymin=0,ymax=80,shape='Square')
    app._tab8_batch_busy=False
    tables=core._tab8_complete_export_tables(app)
    check(len(tables['FrameSummary'])==2 and len(tables['DFitEdges'])==8,
          'Tab 8 rectangular exports retain every frame and radial edge')
    check(set(tables['DFitEdges'].Frame)=={1,3} and set(tables['DFitEdges'].Frame_Index)=={0,2},
          'Tab 8 angular tables carry explicit frame/index identity')
    with patch.object(core.filedialog,'asksaveasfilename',return_value=str(output/'blob.csv')), \
         patch.object(app,'_export_initialdir',return_value=str(output)), \
         patch.object(core,'_export_native_open',return_value=True):
        check(core._tab8_v7_export_table(app),'Tab 8 summary export succeeds')
    saved=pd.read_csv(output/'blob.csv');edges=pd.read_csv(output/'blob_DFitEdges.csv')
    check(np.allclose(saved['Avg. Dia (nm)'],[120.123456789,110.987654321],rtol=1e-14),
          'Tab 8 export retains full stored measurement precision')
    check(len(edges)==8 and set(edges.Frame)=={1,3},'Tab 8 export includes radial measurements from all stored frames')
    app._tab8_frame_var.set(0)
    with patch.object(core.filedialog,'asksaveasfilename',return_value=str(output/'graphs.csv')), \
         patch.object(app,'_export_initialdir',return_value=str(output)), \
         patch.object(core,'_export_native_open',return_value=True):
        check(core._tab8_v7_export_figure5_csv(app),'Tab 8 graph export succeeds')
    for label in tables:
        saved=pd.read_csv(output/f'graphs_{label}.csv')
        check(list(saved.columns)==list(tables[label].columns) and len(saved)==len(tables[label]),
              'Tab 8 graph companion is a rectangular CSV: '+label)
    # Display precision is intentionally distinct from numerical export precision.
    tree=app._lpa_measure_tree
    for child in tree.get_children():tree.delete(child)
    columns=list(tree['columns'])
    tree.insert('','end',values=['test']+['1.23457']*(len(columns)-1))
    app._lpa_gaussian_results={0:dict(fwhm_nm=1.23456789012345,fwhm_fit_sd_nm=.01234567890123)}
    app._lpa_measurements=[dict(line=0,m1_nm=1.123456789012,m2_nm=4.567890123456,delta_nm=3.444433334444)]
    with patch.object(core.filedialog,'asksaveasfilename',return_value=str(output/'line.csv')), \
         patch.object(app,'_export_initialdir',return_value=str(output)), \
         patch.object(core,'_export_native_open',return_value=True):
        core._t7_export_all_visible_table(app)
    full=pd.read_csv(output/'line_Gaussian_FullPrecision.csv')
    endpoint=pd.read_csv(output/'line_Endpoints_FullPrecision.csv')
    check(np.isclose(full.fwhm_nm.iloc[0],1.23456789012345,rtol=1e-14),
          'Tab 7 Gaussian companion retains precision beyond visible rounded text')
    check(np.isclose(endpoint.m1_nm.iloc[0],1.123456789012,rtol=1e-14) and 'Frame_Index' in endpoint,
          'Tab 7 endpoint companion retains precision and frame identity')
    # HDF5 and FRC reports are data-driven exports and must retain their units and samples.
    sample=output/'fixture.h5'
    with core.h5py.File(sample,'w') as h5:
        h5['reconstruction']=(xx+1j*yy).astype(complex)
        h5['signal']=np.arange(5,dtype=float);h5['signal'].attrs['units']='a.u.'
    app.tab12_file_path=sample;app.tab12_frc_entries=[]
    app.tab13_frc_file_path=sample
    app.tab13_frc_entries=[dict(name='fixture',path='/frc',shape=(3,),dtype='float64',
        magnet_B=-7.125,resolution=4.,q_cut=.25,FFT_pixel_cut=2.,dataset_attrs={},group_attrs={},
        q=np.array([0.,.5,1.]),frc=np.array([1.,.6,.2]))]
    for name,filename in [('_tab12_export_report','hdf.json'),('_tab12_export_inventory_csv','hdf.csv'),
                          ('_tab13_export_report','frc.json'),('_tab13_export_summary_csv','frc.csv'),
                          ('_tab13_export_plot','frc_samples.csv'),('_tab14_export','log.txt')]:
        with patch.object(core.filedialog,'asksaveasfilename',return_value=str(output/filename)), \
             patch.object(core,'_export_native_open',return_value=True):
            getattr(core,name)(app)
        check((output/filename).stat().st_size>0,'Stored HDF5/FRC/log export writes '+filename)
    check(len(json.loads((output/'hdf.json').read_text())['datasets'])==2,
          'HDF5 JSON includes all datasets and their metadata')
    frc_table=pd.read_csv(output/'frc_samples.csv')
    check(len(frc_table[frc_table.Plot=='FRC Curves'])==3 and
          np.allclose(frc_table.loc[frc_table.Plot=='FRC Curves','FRC'],[1.,.6,.2]),
          'FRC numerical export retains every source sample')
    # A session snapshot must succeed even if some tabs have no measurements.
    with patch.object(core.filedialog,'askdirectory',return_value=str(output)), \
         patch.object(core,'_export_native_open',return_value=True), \
         patch.object(app,'_fit_fixed_center_1d_gaussian',side_effect=AssertionError('Export must not run fitting')):
        folder=app.export_all_measurement_tables()
    check(bool(folder),'All-measurement session export succeeds without fitting')
    manifest=pd.read_csv(folder/'Export_Manifest.csv')
    check(not manifest.Status.str.startswith('Unavailable:').any(),
          'Session manifest has no failed table exporters: '+str(manifest[manifest.Status.str.startswith('Unavailable:')].to_dict('records')))
    for row in manifest[manifest.Status=='Exported'].itertuples():
        saved=pd.read_csv(folder/row.File)
        check(len(saved)==row.Rows,'Session manifest row count matches '+row.File)
    context=pd.read_csv(folder/'Session_Calibration_ROI_Processing.csv')
    check('Finalized pixel scale' in set(context.Quantity),'Session export includes finalized physical pixel scale')
    check(not json.loads((folder/'README.json').read_text())['failures'],'Session metadata reports no hidden failures')
    check(app.export_all_tables_button.cget('text')=='EXPORT ALL MEASUREMENT TABLES',
          'Session export is available from Tab 1')
