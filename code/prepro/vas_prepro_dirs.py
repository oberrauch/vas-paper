"""

"""
# build-ins
import os
import logging

# external libraries
import geopandas as gpd

# local/oggm imports
from oggm import cfg, utils, workflow
import oggm_vas as vascaling

if __name__ == '__main__':
    # Initialize OGGM and set up the default run parameters
    vascaling.initialize(logging_level='WORKFLOW')
    rgi_version = '62'
    cfg.PARAMS['border'] = 80

    # CLUSTER paths
    wdir = os.environ.get('WORKDIR', '')
    cfg.PATHS['working_dir'] = wdir
    outdir = os.environ.get('OUTDIR', '')

    # define the baseline climate CRU
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
    rgi_reg = os.environ.get('OGGM_RGI_REG', '05')
    if rgi_reg not in ['{:02d}'.format(r) for r in range(1, 20)]:
        raise RuntimeError('Need an RGI Region')
    rgi_ids = gpd.read_file(
        utils.get_rgi_region_file(rgi_reg, version=rgi_version))

    # For greenland we omit connectivity level 2
    if rgi_ids == '05':
        rgi_ids = rgi_ids.loc[rgi_ids['Connect'] != 2]

    # get and set path to intersect shapefile
    intersects_db = utils.get_rgi_intersects_region_file(region=rgi_reg)
    cfg.set_intersects_db(intersects_db)

    # operational run, all glaciers should run
    cfg.PARAMS['use_multiprocessing'] = True
    cfg.PARAMS['continue_on_error'] = True

    # Module logger
    log = logging.getLogger(__name__)
    log.workflow('Starting run for RGI reg {}'.format(rgi_reg))

    # Go - get the pre-processed glacier directories
    base_url = "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/" \
               "L3-L5_files/CRU/elev_bands/qc3/pcp2.5/match_geod"

    gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=3,
                                              prepro_base_url=base_url,
                                              prepro_rgi_version=rgi_version)

    # run vascaling climate tasks
    workflow.execute_entity_task(vascaling.local_t_star, gdirs)
    # adjust mass balance residual with geodetic observations
    vascaling.match_regional_geodetic_mb(gdirs=gdirs, rgi_reg=rgi_reg)
    # prepare historic "spinup"
    workflow.execute_entity_task(vascaling.run_historic_from_climate_data,
                                 gdirs, ys=2000, ye=2020,
                                 output_filesuffix='_historical')
    # store summary
    outpath = os.path.join(wdir, f'run_output_historical_{rgi_reg}.nc')
    utils.compile_run_output(gdirs, input_filesuffix='_historical',
                             path=outpath)

    # compress all gdirs
    workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False)
    # compress 1000 bundles
    utils.base_dir_to_tar()

    log.workflow('OGGM Done')
