#!/bin/bash
#
# Run the mass balance calibration with CRU climate data.
# The resulting files ref_stars.csv and mb_calib_params.csv are stored
# in the ~/run_output/mb_calibration_vas/ directory.
#
#SBATCH --job-name=mb_calibration_vas
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --mail-user=moritz.oberrauch@student.uibk.ac.at
#SBATCH --mail-type=ALL

# Abort whenever a single step fails.
# Without this, bash will just continue on errors.
set -e

# On every node, when slurm starts a job, it will make sure the directory
# /work/username exists and is writable by the jobs user.
# We create a sub-directory there for this job to store its runtime data at.
WORKDIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/mb_calibration_vas/"
mkdir -p "$WORKDIR"
echo "Workdir for this run: $WORKDIR"

# Similarly, let's create a dedicated output directory
OUTDIR="/home/users/moberrauch/paper/run_output/mb_calibration_vas/"
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
  pip install --no-deps "git+https://github.com/OGGM/oggm-vas.git@cfcfac3b1cc5da3c7760dc25b8433c54732d1b7f
"
  # Finally, the run
  python3 mb_calibration_vas.py
EOF

# Copy files from working directory to output directory
cp "$WORKDIR"/*.csv "$OUTDIR" 2>/dev/null || :
cp "$WORKDIR"/*.json "$OUTDIR" 2>/dev/null || :

# Print a final message so you can actually see it being done in the output log.
echo "SLURM DONE"
