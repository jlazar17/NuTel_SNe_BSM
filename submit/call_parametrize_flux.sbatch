#!/bin/bash
#SBATCH -J parametrize_flux
#SBATCH -c 1 # Number of cores
#SBATCH -p arguelles_delgado,shared
#SBATCH --mem 4000
#SBATCH -t 0-02:00:00 # Maximum execution time (D-HH:MM:SS)
#SBATCH -o /n/holyscratch01/arguelles_delgado_lab/Everyone/jlazar/SNe_BSM/output/%A.out # Standard output
#SBATCH -e /n/holyscratch01/arguelles_delgado_lab/Everyone/jlazar/SNe_BSM/error/%A.err # Standard output

eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/setup.sh`
cd /n/home12/jlazar/NuTel_SNe_BSM/
poetry shell
cd scripts
source env.sh
PYTHONPATH=$PYTHONPATH:/n/home12/jlazar/.cache/pypoetry/virtualenvs/sne-bsm-CmpHp_nm-py3.10/lib/python3.10/site-packages/

python 0_parameterize_flux.py --infile ${INFILE} --outfile ${OUTFILE} --thin ${THIN}
