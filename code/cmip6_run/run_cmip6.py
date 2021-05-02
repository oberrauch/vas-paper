""" TODO

"""
# build-ins
import os
import sys
import logging

# externals
import xarray as xr
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# local/oggm modules
from oggm import cfg, utils, workflow, tasks
from oggm.core import gcm_climate
import oggm_vas as vascaling


def run_cmip():
    """



    """
    # Initialize OGGM and set up the default run parameters
    vascaling.initialize(logging_level='DEBUG')
    rgi_version = '62'
    cfg.PARAMS['border'] = 80

    # CLUSTER paths
    wdir = os.environ.get('WORKDIR', '')
    cfg.PATHS['working_dir'] = wdir
    outdir = os.environ.get('OUTDIR', '')

    # define the baseline climate CRU or HISTALP
    cfg.PARAMS['baseline_climate'] = 'CRU'
    # set the mb hyper parameters accordingly TODO
    cfg.PARAMS['prcp_scaling_factor'] = 3
    cfg.PARAMS['temp_melt'] = 0
    cfg.PARAMS['temp_all_solid'] = 4
    cfg.PARAMS['run_mb_calibration'] = False
    # set minimum ice thickness to include in glacier length computation
    # this reduces weird spikes in length records
    cfg.PARAMS['min_ice_thick_for_length'] = 0.1

    # the bias is defined to be zero during the calibration process,
    # which is why we don't use it here to reproduce the results
    cfg.PARAMS['use_bias_for_run'] = True

    # read RGI entry for the glaciers as DataFrame
    # containing the outline area as shapefile
    # RGI glaciers
    rgi_reg = os.environ.get('RGI_REG', '')
    if rgi_reg not in ['{:02d}'.format(r) for r in range(1, 20)]:
        raise RuntimeError('Need an RGI Region')
    rgi_ids = gpd.read_file(
        utils.get_rgi_region_file(rgi_reg, version=rgi_version))

    # get and set path to intersect shapefile
    intersects_db = utils.get_rgi_intersects_region_file(region=rgi_reg)
    cfg.set_intersects_db(intersects_db)

    # operational run, all glaciers should run
    cfg.PARAMS['continue_on_error'] = True 

    # Module logger
    log = logging.getLogger(__name__)
    log.workflow('Starting run for RGI reg {}'.format(rgi_reg))

    # Go - get the pre-processed glacier directories
    base_url = 'https://cluster.klima.uni-bremen.de/~moberrauch/prepro/'
    gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=3,
                                              prepro_base_url=base_url,
                                              prepro_rgi_version=rgi_version)

    # read gcm list
    gcms = pd.read_csv('/home/www/oggm/cmip6/all_gcm_list.csv', index_col=0)

    # iterate over all specified GCMs
    for gcm in sys.argv[1:]:
        # iterate over all SSPs (Shared Socioeconomic Pathways)
        df1 = gcms.loc[gcms.gcm == gcm]
        for ssp in df1.ssp.unique():
            df2 = df1.loc[df1.ssp == ssp]
            assert len(df2) == 2
            # get temperature projections
            ft = df2.loc[df2['var'] == 'tas'].iloc[0]
            # get precipitation projections
            fp = df2.loc[df2['var'] == 'pr'].iloc[0].path
            rid = ft.fname.replace('_r1i1p1f1_tas.nc', '')
            ft = ft.path

            log.workflow('Starting run for {}'.format(rid))

            workflow.execute_entity_task(gcm_climate.process_cmip_data, gdirs,
                                         filesuffix='_' + rid,
                                         # recognize the climate file for later
                                         fpath_temp=ft,
                                         # temperature projections
                                         fpath_precip=fp,  # precip projections
                                         year_range=('1981', '2020'))

            workflow.execute_entity_task(vascaling.run_from_climate_data,
                                         gdirs, climate_filename='gcm_data',
                                         climate_input_filesuffix='_' + rid,
                                         init_model_filesuffix='_historical',
                                         output_filesuffix=rid,
                                         return_value=False)
            gcm_dir = os.path.join(outdir, 'RGI' + rgi_reg, gcm)
            utils.mkdir(gcm_dir)
            utils.compile_run_output(gdirs, input_filesuffix=rid,
                                     path=os.path.join(gcm_dir, rid + '.nc'))

    log.workflow('OGGM Done')


if __name__ == '__main__':
    run_cmip()

