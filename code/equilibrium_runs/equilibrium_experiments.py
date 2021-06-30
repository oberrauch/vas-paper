"""Perform equilibrium experiments with the VAS model.

- the RGI region must be given as environmental variable (handled by slurm)
- `ConstantMassBalance` model using the climate between 1999 and 2019
- Temperature bias between -0.5 degC and 3 degC in 0.5 degC steps

"""
# build-ins
import os
import logging

# externals
import numpy as np
import geopandas as gpd

# local/oggm modules
from oggm import cfg, utils, workflow
import oggm_vas as vascaling

if __name__ == '__main__':
    # initialize OGGM and set up the default run parameters
    vascaling.initialize(logging_level='DEBUG')
    rgi_version = '62'
    cfg.PARAMS['border'] = 80

    # get paths from environmental variables
    # TODO DEBUG: remoce default wdir/outdir
    wdir = os.environ.get('WORKDIR', '/Users/oberrauch/work/paper/workdirs/eq_runs')
    cfg.PATHS['working_dir'] = wdir
    outdir = os.environ.get('OUTDIR', '/Users/oberrauch/work/paper/outdir/eq_runs')

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
    # TODO DEBUG: cfg.PARAMS['use_multiprocessing'] = True
    cfg.PARAMS['use_multiprocessing'] = False
    cfg.PARAMS['use_bias_for_run'] = True

    # read RGI entry for the glaciers as DataFrame
    # containing the outline area as shapefile
    # RGI glaciers
    # TODO DEBUG
    rgi_reg = os.environ.get('RGI_REG', '11')
    if rgi_reg not in ['{:02d}'.format(r) for r in range(1, 20)]:
        raise RuntimeError('Need an RGI Region')
    rgi_ids = gpd.read_file(
        utils.get_rgi_region_file(rgi_reg, version=rgi_version))

    # For greenland we omit connectivity level 2
    if rgi_reg == '05':
        rgi_ids = rgi_ids.loc[rgi_ids['Connect'] != 2]

    # get and set path to intersect shapefile
    intersects_db = utils.get_rgi_intersects_region_file(region=rgi_reg)
    cfg.set_intersects_db(intersects_db)

    # operational run, all glaciers should run
    # TODO DEBUG cfg.PARAMS['continue_on_error'] = True
    cfg.PARAMS['continue_on_error'] = False

    # module logger
    log = logging.getLogger(__name__)
    log.workflow('Starting run for RGI reg {}'.format(rgi_reg))

    # get the pre-processed glacier directories
    base_url = 'https://cluster.klima.uni-bremen.de/' \
               '~moberrauch/prepro_vas_paper/'
    # TODO DEBUG: delete rgi_ids
    rgi_ids = ['RGI60-11.00007']
    gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=3,
                                              prepro_base_url=base_url,
                                              prepro_rgi_version=rgi_version)

    # run constant climate scenario with different temperature biases
    # TODO DEBUG np.arange(..., 5.5, ...):
    for temp_bias in np.arange(-0.5, 1, 0.5):
        filesuffix = "bias{:+.1f}".format(temp_bias)
        # start model run
        workflow.execute_entity_task(vascaling.run_constant_climate,
                                     gdirs, nyears=3000, y0=2009, halfsize=10,
                                     temperature_bias=temp_bias,
                                     init_model_filesuffix='_historical',
                                     output_filesuffix=filesuffix,
                                     return_value=False)
        # compile and store run output
        eq_dir = os.path.join(outdir, 'RGI' + rgi_reg)
        utils.mkdir(eq_dir)
        utils.compile_run_output(gdirs, input_filesuffix=filesuffix,
                                 path=os.path.join(eq_dir, filesuffix + '.nc'))

    log.workflow('OGGM Done')
