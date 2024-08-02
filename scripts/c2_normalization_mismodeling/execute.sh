export OUTFILE1=${DATADIR}/majoron_trials_mismodeling_${MISMODELING_COEFFICIENT}.h5
export BSM_FILE=${DATADIR}/serialized_majoron_fluxes.h5
export SM_FILE=${DATADIR}/serialized_sm_flux.h5
export SM_NAME=sm_flux_0
export N=50000
export SEED=1211

export MISMODELING_COEFFICIENT=$1

for A in `ls ${DATADIR}*.npy | xargs -n 1 basename`
do
    L=${#A}
    BSM_NAME=${A:0:$((L - 4))}_0
    CMD="sbatch -D $PWD --export=SEED=$SEED,BSM_NAME=$BSM_NAME,BSM_FILE=$BSM_FILE,SM_FILE=$SM_FILE,SM_NAME=$SM_NAME,OUTFILE=$OUTFILE1,N=$N,MISMODELING_COEFFICIENT=$MISMODELING_COEFFICIENT submit_run_trials.sbatch"
    $CMD
    SEED=$(($SEED+1))
done
