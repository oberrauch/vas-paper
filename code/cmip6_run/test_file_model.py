""" TODO

"""
# build-ins
import os
import sys
import logging

# externals
import pandas as pd
import geopandas as gpd

# local/oggm modules
from oggm import cfg, utils, workflow
from oggm.core import gcm_climate
import oggm_vas as vascaling

# Initialize OGGM and set up the default run parameters
vascaling.initialize(logging_level='DEBUG')
rgi_version = '62'
cfg.PARAMS['border'] = 80

# CLUSTER paths
wdir = utils.gettempdir('fmod_wdir')
utils.mkdir(wdir)
cfg.PATHS['working_dir'] = wdir
outdir = utils.gettempdir('fmod_outdir_old')
utils.mkdir(outdir)

# define the baseline climate CRU or HISTALP
cfg.PARAMS['baseline_climate'] = 'CRU'
# set the mb hyper parameters accordingly
cfg.PARAMS['prcp_scaling_factor'] = 3
cfg.PARAMS['temp_melt'] = 0
cfg.PARAMS['temp_all_solid'] = 4
cfg.PARAMS['prcp_default_gradient'] = 4e-4
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
rgi_reg = '11'
rgi_ids = gpd.read_file(
    utils.get_rgi_region_file(rgi_reg, version=rgi_version))
rgi_ids = ['RGI60-11.00897']

# get and set path to intersect shapefile
intersects_db = utils.get_rgi_intersects_region_file(region=rgi_reg)
cfg.set_intersects_db(intersects_db)

# operational run, all glaciers should run
cfg.PARAMS['continue_on_error'] = False
cfg.PARAMS['use_multiprocessing'] = False

# Module logger
log = logging.getLogger(__name__)
log.workflow('Starting run for RGI reg {}'.format(rgi_reg))

# Go - get the pre-processed glacier directories
base_url = 'https://cluster.klima.uni-bremen.de/' \
           '~moberrauch/prepro_vas_paper/'
gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=3,
                                          prepro_base_url=base_url,
                                          prepro_rgi_version=rgi_version)

# get temperature projections
dir_path = '/Users/oberrauch/work/paper/data/CESM2'
ft = os.path.join(dir_path, 'CESM2_ssp585_r1i1p1f1_tas.nc')
# get precipitation projections
fp = os.path.join(dir_path, 'CESM2_ssp585_r1i1p1f1_pr.nc')
rid = 'CESM2_ssp585'

log.workflow('Starting run for {}'.format(rid))

workflow.execute_entity_task(gcm_climate.process_cmip_data, gdirs,
                             # recognize the climate file for later
                             filesuffix='_' + rid,
                             # temperature projections
                             fpath_temp=ft,
                             # precip projections
                             fpath_precip=fp,
                             year_range=('1981', '2020'))

workflow.execute_entity_task(vascaling.run_from_climate_data,
                             gdirs,
                             # use gcm_data, not climate_historical
                             climate_filename='gcm_data',
                             # use a different scenario
                             climate_input_filesuffix='_' + rid,
                             # this is important! Start from 2019
                             init_model_filesuffix='_historical',
                             # recognize the run for later
                             output_filesuffix='_' + rid,
                             return_value=False)
gcm_dir = os.path.join(outdir, 'RGI' + rgi_reg)
utils.mkdir(gcm_dir)
utils.compile_run_output(gdirs, input_filesuffix='_' + rid,
                         path=os.path.join(rid + '.nc'))

log.workflow('OGGM Done')
