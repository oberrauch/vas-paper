#!/bin/bash

# Iterate over all general circulation models (GCM) and call the

GCMS="CESM2-WACCM MPI-ESM1-2-HR GFDL-ESM4 NorESM2-MM INM-CM4-8 INM-CM5-0 MRI-ESM2-0 CESM2 EC-Earth3 EC-Earth3-Veg CAMS-CSM1-0 BCC-CSM2-MR FGOALS-f3-L TaiESM1 CMCC-CM2-SR5"

for GCM in $GCMS; do
	sbatch --array=11 run_cmip6.slurm $GCM
done
