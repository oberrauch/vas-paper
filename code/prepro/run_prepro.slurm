#!/bin/bash
#
#SBATCH --job-name=vas_prepro
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --mail-user=moritz.oberrauch@student.uibk.ac.at
#SBATCH --mail-type=ALL
#SBATCH --qos=low

# Abort whenever a single step fails.
# Without this, bash will just continue on errors.
set -e

# Current RGI region
OGGM_RGI_REG=`printf "%02d" $SLURM_ARRAY_TASK_ID`
export OGGM_RGI_REG
echo "RGI Region: $OGGM_RGI_REG"

# On every node, when slurm starts a job, it will make sure the directory
# /work/username exists and is writable by the jobs user.
# We create a sub-directory there for this job to store its runtime data at.
WORKDIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/vas_prepro"
mkdir -p "$WORKDIR"
echo "Workdir for this run: $WORKDIR"

# The preprocessed glacier directories are directly stored in the publicly
# available directory on the cluster.
OUTDIR="/home/www/moberrauch/prepro_vas_paper"
mkdir -p "$OUTDIR"
echo "Outdir for this run: $OUTDIR"

# Export the WORKDIR and OUTDIR as environment variable
# so our script can use it to find its working directory.
export WORKDIR
export OUTDIR

# Use the local data download cache
export OGGM_DOWNLOAD_CACHE=/home/data/download
export OGGM_DOWNLOAD_CACHE_RO=1
export OGGM_EXTRACT_DIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/oggm_tmp"

# All commands in the EOF block run inside of the container. Adjust container
# version to your needs, they are guaranteed to never change after their
# respective day has passed.
srun -u -n 1 -c "${SLURM_JOB_CPUS_PER_NODE}" singularity exec /home/users/moberrauch/oggm_20210430.sif bash -s <<EOF
  set -e
  # Setup a fake home dir inside of our workdir, so we don't clutter the actual
  # shared homedir with potentially incompatible stuff.
  export HOME="$WORKDIR/fake_home"
  mkdir "\$HOME"
  # Create a venv that _does_ use system-site-packages, since everything is
  # already installed on the container. We cannot work on the container itself,
  # as the base system is immutable.
  python3 -m venv --system-site-packages "$WORKDIR/oggm_env"
  source "$WORKDIR/oggm_env/bin/activate"
  # Make sure latest pip is installed
  pip install --upgrade pip setuptools
  # Install needed dependencies
  pip install sklearn
  # OPTIONAL: install OGGM latest and OGGM-VAS
  pip install --no-deps "git+https://github.com/OGGM/oggm.git@7cfc96c1c0c4c3951cb761c81b28f9a0caea4187"
  pip install --no-deps "git+https://github.com/OGGM/oggm-vas.git@c994628e544e45b010b489d47614418059e775bd"
  # Finally, the run
  python3 vas_prepro_dirs.py
EOF

# Write out
echo "Copying files..."
rsync -avzh "$WORKDIR/run_output_historical_$OGGM_RGI_REG.nc" "$OUTDIR"
# Adjust directory structure according to OGGM prepro dir convention.
# The directory structure RGI62/b_080/L3 corresponds to RGI version 6.2,
# cfp.PARAMS['border'] = 80 and prepro level 3
rsync -avzh "$WORKDIR/per_glacier/"* "$OUTDIR/RGI62/b_080/L3/"

# Print a final message so you can actually see it being done in the output log.
echo "SLURM DONE"
