#!/bin/bash
#SBATCH -J run_trials
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

TMPFILE="/n/holyscratch01/arguelles_delgado_lab/Everyone/jlazar/SNe_BSM/.tmp_${SLURM_JOB_ID}.txt"

if [ -z ${SM_UNCERTAINTY+x} ]; then
    SM_UNCERTAINTY=0.2
fi

if [ -z ${MISMODELING_COEFFICIENT+x} ]; then
    MISMODELING_COEFFICIENT=1.0
fi

CMD="python 1_run_trials.py --bsm_file $BSM_FILE --bsm_name $BSM_NAME --sm_file $SM_FILE --sm_name $SM_NAME -n $N --outfile $OUTFILE --no_track --tmpfile $TMPFILE --mismodeling_coefficient $MISMODELING_COEFFICIENT --sm_uncertainty $SM_UNCERTAINTY --seed $SEED"
echo $CMD
$CMD
