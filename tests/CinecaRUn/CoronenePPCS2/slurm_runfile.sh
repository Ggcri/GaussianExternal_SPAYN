#!/bin/bash
#SBATCH -J Cor
#SBATCH -p dcgp_usr_prod
#SBATCH -A CNHPC_1491856
#SBATCH -N 1
#SBATCH --ntasks-per-node=110
#SBATCH --cpus-per-task=1
#SBATCH --mem=440Gb
#SBATCH --time=24:00:00
#SBATCH --gres=tmpfs:3t
set +e

# --- Moduli ---
module purge
module load profile/chem-phys
module load g16
. $g16root/g16/bsd/g16.profile
module load Molpro/2025.3            
module load External/LoadExternal.module

# --- Variabili d'ambiente ---
export SCRATCH="/tmp/mol_$SLURM_JOBID"
export TMPDIR=$SCRATCH
export GAUSS_SCRDIR="/tmp/g16_$SLURM_JOBID"


mkdir -p $SCRATCH
mkdir -p $GAUSS_SCRDIR
echo "Scratch dir: $SCRATCH"

cd $SLURM_SUBMIT_DIR
touch JOB_ID_${SLURM_JOB_ID}
touch NowRunning_Single

g16 Coronene.gjf

rm -f NowRunning_Single
touch Completed_Single
rm -rf $SCRATCH
rm -rf $GAUSS_SCRDIR
