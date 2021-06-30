# -------------------------------------------------------------------
# Computing relative volume change and applying it to missing volume
# -------------------------------------------------------------------
log.info('Computing relative volume change '
         'and applying it to missing volume')

# iterate over all *.nc files
nc = all_ncs[12]
print(nc)


# get SSP and GCM descriptions to create a dataset key
ssp = os.path.basename(nc).split('_')[1].replace('.nc', '')
gcm = os.path.basename(nc).split('_')[0]
key_both = ssp + '_' + gcm

# open the dataset
ds_normal =  xr.open_dataset(nc)
# compute initial regional aggregate volume
v0_normal = ds_normal.volume.isel(time=0)
v0_region_normal = v0_normal.sum(dim='rgi_id')
# compute cumulative aggregrate regional volume difference
dv_cum_region_normal = ds_normal.volume.sum(dim='rgi_id').diff(
    dim='time').cumsum()
# apply to volume of missing glaciers
v0_missing_normal = missing_volume.loc[
    failed_dict[key_both]].sum()
vol_ts_missing_normal = v0_missing_normal * (1 + dv_cum_region_normal / v0_region_normal)
# compute aggregate volume time series including missing volume
regional_volume_corrected_normal = ds_normal.volume.sum(
    dim='rgi_id') + vol_ts_missing_normal
# add initial value
regional_volume_corrected_normal = xr.concat(
    [v0_region_normal + v0_missing_normal, regional_volume_corrected_normal],
    dim='time')