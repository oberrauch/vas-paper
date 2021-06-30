"""Correct volume for missing glaciers

Some glaciers fail during the global/regional runs due to different issues.
Therefore, those glaciers and their ice volume are missing from the aggregate value
which skews the absolute result. This scripts accounts for the missing ice volume and its
change by (1.) getting the initial volume and (2.) estimating the final volume (or volume
change) by applying the regional average relative change to the found initial volume.

The initial volume is found either by (a) getting the initial volume from another run (with a
different temperature bias) where the glacier did not fail or, (b) getting the initial volume
using volume/area scaling from the RGI area with regionally calibrated scaling parameters (for
glaciers which failed for).

TODO: finish docstring

The new dataset is stored under ...

"""

# build-ins
import os
import logging

# external libs
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt

# local and oggm modules
from oggm import utils

# instance logger
log = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=getattr(logging, 'INFO'))


def correct_eq_volume(rgi_reg, vas=True, log_file_path='logfile.txt'):
    """Corrects aggregate regional equilibrium volume for missing glaciers."""

    # -----------------------------------------------------------
    # Reading model output files and collecting missing glaciers
    # -----------------------------------------------------------

    # read model output files
    log.info('Reading model output files and collecting missing glaciers')

    # define path to input and output directory
    if vas:
        dir_path = '/home/users/moberrauch/run_output/eq_runs'
    else:
        dir_path = '/home/www/fmaussion/vas_paper/run_equi/equi_output'
        
    out_path = '/home/users/moberrauch/run_output/eq_runs/'
    

    # create empty containers
    missing_glaciers_per_bias = dict()

    # iterate over all temperature biases
    for temp_bias in np.arange(-0.5, 5.5, 0.5):
        # load entire dataset
        f_name = f'bias{temp_bias:+.1f}.nc'
        f_path = os.path.join(dir_path, f'RGI{rgi_reg:02d}', f_name)
        ds = xr.load_dataset(f_path)
        # get 'missing' glaciers
        missing_rgi_ids = ds.rgi_id.where(np.isnan(ds.volume.isel(time=0)))
        missing_rgi_ids = missing_rgi_ids.dropna(dim='rgi_id').values
        # add to containers
        missing_glaciers_per_bias[temp_bias] = missing_rgi_ids

    # get unique ids
    missing_rgi_ids = sorted(
        {a for b in missing_glaciers_per_bias.values() for a in b})
    # use as index for a DataFrame
    missing_volume = pd.Series(index=missing_rgi_ids, dtype=float)

    # ---------------------------------------------------------------
    # Searching for initial volume of missing glaciers in other runs
    # ---------------------------------------------------------------

    log.info('Searching for initial volume of missing glaciers in other runs')

    # iterate over all temperature biases
    for temp_bias in np.arange(-0.5, 5.5, 0.5):
        # load entire dataset
        f_name = f'bias{temp_bias:+.1f}.nc'
        f_path = os.path.join(dir_path, f'RGI{rgi_reg:02d}', f_name)
        ds = xr.load_dataset(f_path)

        # get initial volume for missing glaciers
        init_volume = ds.volume.where(ds.rgi_id.isin(missing_rgi_ids))
        init_volume = init_volume.isel(time=0).dropna(dim='rgi_id')
        init_volume = init_volume.to_pandas()

        # fill nan with new values (if exist)
        missing_volume.update(init_volume)

        # break if all initial volumes are found
        if not missing_volume.isna().any():
            break
            
    # delete glacier with "missing" volume that vanished before 2020
    # this is a quick and dirty bugfix and can be removed, once the
    # equilibrium runs are re-done
    missing_volume = missing_volume[missing_volume != 0]
    missing_rgi_ids = missing_volume.index.values
            
    # --------------------------------------------------------
    # Compile statistics of missing glaciers and write to log
    # --------------------------------------------------------
    # TODO: this is a quick and dirty method, but it does the trick

    log.info('Compile statistics of missing glaciers and write to log file.')

    # write number and area fraction of missing glaciers to file
    f = open(log_file_path, 'a')
    # read glacier statistics for given rgi region
    summary_dir_path = '/home/www/oggm/gdirs/oggm_v1.4/L3-L5_files/CRU/' \
                       'elev_bands/qc3/pcp2.5/match_geod/RGI62/b_080/L3/' \
                       'summary/'
    df = pd.read_csv(os.path.join(summary_dir_path,
                                  f'glacier_statistics_{rgi_reg:02d}.csv'),
                     index_col=0)
    area_fraction = df.loc[
                        missing_rgi_ids].rgi_area_km2.sum() / df.rgi_area_km2.sum()
    f.write(f"{'vas' if vas else 'fl'}\t{rgi_reg:02d}"
            f"\t{temp_bias:+1.1f}\t{len(missing_rgi_ids)}"
            f"\t{len(ds.rgi_id)}\t{area_fraction:.4f}\n")
    f.close()
    
    return

    # -----------------------------------------------------------
    # Computing still missing initial volumes via scaling method
    # -----------------------------------------------------------

    # fill still missing initial volumes using scaling relation
    if missing_volume.isna().any():

        log.info('Computing still missing initial volumes via scaling method')

        # read glacier statistics for given rgi region
        summary_dir_path = '/home/www/oggm/gdirs/oggm_v1.4/L3-L5_files/CRU/' \
                           + 'elev_bands/qc3/pcp2.5/match_geod/RGI62/b_080/L3/summary/'
        df = pd.read_csv(os.path.join(summary_dir_path,
                                      f'glacier_statistics_{rgi_reg:02d}.csv'),
                         index_col=0)

        # distinguish between glaciers and ice caps
        scaling_params = [[0.0340, 1.375], [0.0538, 1.25]]
        scaling_params = pd.DataFrame(scaling_params,
                                      index=['Glacier', 'Ice cap'],
                                      columns=['constant', 'exponent'])
        for glacier_type in ['Glacier', 'Ice cap']:
            df_type = df[df.glacier_type == glacier_type]
            if df_type.empty:
                continue
            # get scaling parameters (lin reg in log/log space)
            dfl = np.log(df_type[['inv_volume_km3', 'rgi_area_km2']])
            dfl = dfl.dropna()
            try:
                regr_result = stats.linregress(dfl.rgi_area_km2.values,
                                               dfl.inv_volume_km3.values)
                slope, intercept, r_value, p_value, std_err = regr_result
                scaling_params.loc[glacier_type, 'constant'] = np.exp(
                    intercept)
                scaling_params.loc[glacier_type, 'exponent'] = slope
            except ValueError:
                # use default global values
                pass

        # compute initial volume using scaling relation
        missing_rgi_ids = missing_volume[missing_volume.isna()].index.values
        init_volume = (scaling_params.loc[df.loc[missing_rgi_ids].glacier_type,
                                          'constant'].values *
                       df.loc[missing_rgi_ids].rgi_area_km2 **
                       scaling_params.loc[df.loc[missing_rgi_ids].glacier_type,
                                          'exponent'].values)

        # fill nan with scaled initial volumes, converted to m3 from km3
        missing_volume.update(init_volume * 1e9)

    # -------------------------------------------------------------------
    # Computing relative volume change and applying it to missing volume
    # -------------------------------------------------------------------
    log.info('Computing relative volume change '
             'and applying it to missing volume')

    # get average relative volume change
    dir_path = '/home/users/moberrauch/run_output/eq_runs/'
    f_path = os.path.join(dir_path,
                          f"{'vas' if vas else 'fl'}_volume_rgi_region.nc")
    vol_rgi_region = xr.load_dataset(f_path)
    vol_rgi_region = vol_rgi_region.sel(rgi_reg=rgi_reg).volume

    # create empty DataArray for new volume
    vol_corrected = vol_rgi_region.copy(deep=True)

    # iterate over all temperature biases
    for temp_bias in np.arange(-0.5, 5.5, 0.5):
        # compute relative volume change for all given glaciers
        vol = vol_rgi_region.sel(temp_bias=temp_bias)
        relative_vol_change = float(
            (vol.diff(dim='time') / vol.isel(time=0)).values)
        # apply to volume of missing glaciers
        vol_0 = missing_volume.loc[missing_glaciers_per_bias[temp_bias]].sum()
        vol_1 = vol_0 * (1 + relative_vol_change)
        # add to new DataArray
        vol_corrected.loc[{'temp_bias': temp_bias}] += [vol_0, vol_1]

    # new dataset and store to file
    out_path = os.path.join(out_path, f"{'vas' if vas else 'fl'}_volume"
                                      + f"_corrected_{rgi_reg:02d}.nc")

    # ---------------------------------
    # Storing corrected volume to file
    # ---------------------------------

    log.info(f'Storing corrected volume to file at {out_path}')
    vol_corrected.to_netcdf(out_path)


def correct_volume_timeseries(rgi_reg, vas=True):
    """Corrects aggregate regional equilibrium volume for missing glaciers."""

    # -----------------------------------------------------------
    # Reading model output files and collecting missing glaciers
    # -----------------------------------------------------------

    # read model output files
    log.info('Reading model output files and collecting missing glaciers')

    # define path to input and output directory
    if vas:
        dir_path = '/home/users/moberrauch/run_output/eq_runs'
    else:
        dir_path = '/home/www/fmaussion/vas_paper/run_equi/equi_output'     

    out_path = '/home/users/moberrauch/run_output/eq_runs/'
   
    # create empty containers
    missing_glaciers_per_bias = dict()

    # iterate over all temperature biases
    for temp_bias in np.arange(-0.5, 5.5, 0.5):
        # load entire dataset
        f_name = f'bias{temp_bias:+.1f}.nc'
        f_path = os.path.join(dir_path, f'RGI{rgi_reg:02d}', f_name)
        ds = xr.load_dataset(f_path)
        # get 'missing' glaciers
        missing_rgi_ids = ds.rgi_id.where(np.isnan(ds.volume.isel(time=0)))
        missing_rgi_ids = missing_rgi_ids.dropna(dim='rgi_id').values
        # add to containers
        missing_glaciers_per_bias[temp_bias] = missing_rgi_ids

    # get unique ids
    missing_rgi_ids = sorted(
        {a for b in missing_glaciers_per_bias.values() for a in b})
    # use as index for a DataFrame
    missing_volume = pd.Series(index=missing_rgi_ids, dtype=float)

    # ---------------------------------------------------------------
    # Searching for initial volume of missing glaciers in other runs
    # ---------------------------------------------------------------

    log.info('Searching for initial volume of missing glaciers in other runs')

    # iterate over all temperature biases
    for temp_bias in np.arange(-0.5, 5.5, 0.5):
        # load entire dataset
        f_name = f'bias{temp_bias:+.1f}.nc'
        f_path = os.path.join(dir_path, f'RGI{rgi_reg:02d}', f_name)
        ds = xr.load_dataset(f_path)

        # get initial volume for missing glaciers
        init_volume = ds.volume.where(ds.rgi_id.isin(missing_rgi_ids))
        init_volume = init_volume.isel(time=0).dropna(dim='rgi_id')
        init_volume = init_volume.to_pandas()

        # fill nan with new values (if exist)
        missing_volume.update(init_volume)

        # break if all initial volumes are found
        if not missing_volume.isna().any():
            break

    # -----------------------------------------------------------
    # Computing still missing initial volumes via scaling method
    # -----------------------------------------------------------

    # fill still missing initial volumes using scaling relation
    if missing_volume.isna().any():

        log.info('Computing still missing initial volumes via scaling method')

        # read glacier statistics for given rgi region
        summary_dir_path = '/home/www/oggm/gdirs/oggm_v1.4/L3-L5_files/CRU/' \
                           + 'elev_bands/qc3/pcp2.5/match_geod/RGI62/b_080/L3/summary/'
        df = pd.read_csv(os.path.join(summary_dir_path,
                                      f'glacier_statistics_{rgi_reg:02d}.csv'),
                         index_col=0)

        # distinguish between glaciers and ice caps
        scaling_params = [[0.0340, 1.375], [0.0538, 1.25]]
        scaling_params = pd.DataFrame(scaling_params,
                                      index=['Glacier', 'Ice cap'],
                                      columns=['constant', 'exponent'])
        for glacier_type in ['Glacier', 'Ice cap']:
            df_type = df[df.glacier_type == glacier_type]
            if df_type.empty:
                continue
            # get scaling parameters (lin reg in log/log space)
            dfl = np.log(df_type[['inv_volume_km3', 'rgi_area_km2']])
            dfl = dfl.dropna()
            try:
                regr_result = stats.linregress(dfl.rgi_area_km2.values,
                                               dfl.inv_volume_km3.values)
                slope, intercept, r_value, p_value, std_err = regr_result
                scaling_params.loc[glacier_type, 'constant'] = np.exp(
                    intercept)
                scaling_params.loc[glacier_type, 'exponent'] = slope
            except ValueError:
                # use default global values
                pass

        # compute initial volume using scaling relation
        missing_rgi_ids = missing_volume[missing_volume.isna()].index.values
        init_volume = (scaling_params.loc[df.loc[missing_rgi_ids].glacier_type,
                                          'constant'].values *
                       df.loc[missing_rgi_ids].rgi_area_km2 **
                       scaling_params.loc[df.loc[missing_rgi_ids].glacier_type,
                                          'exponent'].values)

        # fill nan with scaled initial volumes, converted to m3 from km3
        missing_volume.update(init_volume * 1e9)

    # -------------------------------------------------------------------
    # Computing relative volume change and applying it to missing volume
    # -------------------------------------------------------------------
    log.info('Computing relative volume change '
             'and applying it to missing volume')

    # create empty container
    regional_volume_corrected = list()

    # iterate over all temperature biases
    for temp_bias in np.arange(-0.5, 5.5, 0.5):
        # load entire dataset
        f_name = f'bias{temp_bias:+.1f}.nc'
        f_path = os.path.join(dir_path, f'RGI{rgi_reg:02d}', f_name)
        ds = xr.load_dataset(f_path)
        # compute initial regional aggregate volume
        v0 = ds.volume.isel(time=0)
        v0_region = v0.sum(dim='rgi_id')
        # compute cumulative aggregrate regional volume difference
        dv_cum_region = ds.volume.sum(dim='rgi_id').diff(dim='time').cumsum()
        # apply to volume of missing glaciers
        v0_missing = missing_volume.loc[
            missing_glaciers_per_bias[temp_bias]].sum()
        vol_ts_missing = v0_missing * (1 + dv_cum_region / v0_region)
        # compute aggretate regional volume time series including missing volume
        regional_volume_corrected_bias = ds.volume.sum(
            dim='rgi_id') + vol_ts_missing
        # add initial value
        regional_volume_corrected_bias = xr.concat(
            [v0_region + v0_missing, regional_volume_corrected_bias],
            dim='time')
        # add temperature bias as coordinate and add to container
        regional_volume_corrected_bias.coords['temp_bias'] = temp_bias
        regional_volume_corrected.append(regional_volume_corrected_bias)

    # concat into a single dataset along the dimension of temperature bias
    regional_volume_corrected = xr.concat(regional_volume_corrected,
                                          dim='temp_bias')
    # new dataset and store to file
    out_path = os.path.join(out_path, f"{'vas' if vas else 'fl'}_volume_ts_"
                                      + f"corrected_{rgi_reg:02d}.nc")

    # ---------------------------------
    # Storing corrected volume to file
    # ---------------------------------

    log.info(f'Storing corrected volume to file at {out_path}')
    regional_volume_corrected.to_netcdf(out_path)


def run_eq_vol_correction():
    """Read model output for VAS and flowline model
    and correct equilibrium volume for missing glaciers."""
    # path to log file
    log_file_path = '/home/users/moberrauch/paper/missing_glaciers.txt'

    # get RGI region number from slurm array
    rgi_reg = os.environ.get('RGI_REG', '')
    if rgi_reg not in ['{:02d}'.format(r) for r in range(1, 20)]:
        raise RuntimeError('Need an RGI Region')
    rgi_reg = int(rgi_reg)
    # correct vas model data
    log.info('Correcting volume data for VAS model')
    correct_eq_volume(rgi_reg, vas=True, log_file_path=log_file_path)
    # correct flowline model data
    log.info('Correcting volume data for flowline model')
    correct_eq_volume(rgi_reg, vas=False, log_file_path=log_file_path)
    log.info('Done')


def run_vol_timeseries_correction():
    """Read model output for VAS and flowline model
    and correct volume time series for missing glaciers."""

    # get RGI region number from slurm array
    rgi_reg = os.environ.get('RGI_REG', '')
    if rgi_reg not in ['{:02d}'.format(r) for r in range(1, 20)]:
        raise RuntimeError('Need an RGI Region')
    rgi_reg = int(rgi_reg)
    # correct vas model data
    log.info('Correcting volume data for VAS model')
    correct_volume_timeseries(rgi_reg, vas=True)
    # correct flowline model data
    log.info('Correcting volume data for flowline model')
    correct_volume_timeseries(rgi_reg, vas=False)
    log.info('Done')
