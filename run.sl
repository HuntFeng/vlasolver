#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=168:00:00
#SBATCH --mail-user=cuichen@virginia.edu
#SBATCH --mail-type=all
#SBATCH --partition=standard
#SBATCH --constraint=afton
#SBATCH --account=CHEN_CUI
#SBATCH --output=out_Vlasolver.out

WORK_HOME=/scratch/jpz5rz/20240830_ppc100/
cd $WORK_HOME

module load matlab


# source /apps/software/standard/mpi/gcc/11.4.0/openmpi/4.1.4/imkl/2023.1.0/setvars.sh

# export OMPI_MCA_btl='^uct,ofi'
# export OMPI_MCA_pml='ucx'
# export OMPI_MCA_mtl='^ofi'

# cd run/
# sh hpc_run.sh -n $SLURM_NTASKS

matlab -nodisplay -r "pices1d_x_512_1st_order; pices1d_x_512_2nd_order; pices1d_x_512_3rd_order; exit;"
