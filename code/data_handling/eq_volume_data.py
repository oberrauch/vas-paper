"""The equilibrium run(s) produce(s) a lot of data (about 20GB). It is more
convenient to handle the files on the cluster."""

# built-ins
import os
import logging

# external libs
import numpy as np
import xarray as xr

# instance logger
log = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=getattr(logging, 'INFO'))


def get_eq_volume(out_path=None):
    """Get aggregate equilibrium for each RGI region and each temperature bias.

    The new dataset with RGI region number and temperature bias as new
    dimensions is stored to file, the path can be given.
    """

    # define path
    dir_path = '/home/users/moberrauch/run_output/eq_runs'

    ds = list()

    # iterate over all RGI regions
    for rgi_reg in np.arange(1, 19):
        log.info(f'Reading files for RGI region {rgi_reg:d}')
        ds_ = list()
        # iterate over all temperature biases
        for temp_bias in np.arange(-0.5, 3, 0.5):
            # load entire dataset
            f_name = f'bias{temp_bias:+.1f}.nc'
            f_path = os.path.join(dir_path, f'RGI{rgi_reg:02d}', f_name)
            log.info(f'Reading file {f_name}')
            ds__ = xr.load_dataset(f_path)
            # select initial and equilibrium volume
            ds__ = ds__.volume.isel(time=[0, -1]).sum(dim='rgi_id')
            # add temperature bias as coordinate
            ds__.coords['temp_bias'] = temp_bias
            ds_.append(ds__)
        # concat all dataset of one RGI region
        # with temperature bias as new dimension
        ds_ = xr.concat(ds_, dim='temp_bias')
        # add RGI region number as coordinate
        ds_.coords['rgi_reg'] = rgi_reg
        ds.append(ds_)

    # concat into one dataset over RGI region
    log.info('Combining into one dataset')
    ds = xr.concat(ds, dim='rgi_reg')
    # store to file
    if not out_path:
        out_path = os.path.join(dir_path, 'volume_rgi_region.nc')
    log.info(f'Storing new dataset to {out_path}')
    ds.to_netcdf(out_path)

    # return
    return ds


if __name__ == '__main__':
    get_eq_volume()
