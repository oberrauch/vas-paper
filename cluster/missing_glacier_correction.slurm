#!/bin/bash
#
#SBATCH --job-name=volume_correction
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
RGI_REG=`printf "%02d" $SLURM_ARRAY_TASK_ID`
export RGI_REG
echo "RGI Region: $RGI_REG"

# On every node, when slurm starts a job, it will make sure the directory
# /work/username exists and is writable by the jobs user.
# We create a sub-directory there for this job to store its runtime data at.
WORKDIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/volume_correction/"
mkdir -p "$WORKDIR"
echo "Workdir for this run: $WORKDIR"

# Export the WORKDIR and OUTDIR as environment variable
# so our script can use it to find its working directory.
export WORKDIR

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
  # Finally, the run
  python3 -c "from missing_glacier_correction import run_eq_vol_correction; run_eq_vol_correction()"
  # python3 -c "from missing_glacier_correction import run_vol_timeseries_correction; run_vol_timeseries_correction()"
EOF

# Print a final message so you can actually see it being done in the output log.
echo "SLURM DONE"
