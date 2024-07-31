```bash
export OUTFILE1=${DATADIR}/majoron_trials_gen2.h5
export SEED=1
export BSM_FILE=${DATADIR}/serialized_majoron_fluxes.h5
export SM_FILE=${DATADIR}/serialized_sm_flux.h5
export SM_NAME=sm_flux_0
export N=50000
export HIT_SCALING=6.2882088208820885

for A in `ls ${DATADIR}*.npy | xargs -n 1 basename`
do
    L=${#A}
    BSM_NAME=${A:0:$((L - 4))}_0
    CMD="sbatch -D $PWD --export=SEED=$SEED,BSM_NAME=$BSM_NAME,BSM_FILE=$BSM_FILE,SM_FILE=$SM_FILE,SM_NAME=$SM_NAME,OUTFILE=$OUTFILE1,N=$N,HIT_SCALING=$HIT_SCALING submit_run_trials.sbatch"
    $CMD
    SEED=$(($SEED+1))
done
```
