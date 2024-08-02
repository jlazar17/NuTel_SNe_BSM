```bash
export OUTFILE1=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jlazar/SNe_BSM//magnetic_moment_trials.h5
export BSM_FILE=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jlazar/SNe_BSM//serialized_magnetic_moment_fluxes.h5
export SM_FILE=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jlazar/SNe_BSM//serialized_sm_flux.h5
export SM_NAME=sm_flux_0
export N=50000
export SEED=1211
for A in `ls ${DATADIR}/new_fluxes/*nf.npy | xargs -n 1 basename`
do
    L=${#A}
    BSM_NAME=${A:0:$((L - 4))}_0
    CMD="sbatch -D $PWD --export=SEED=$SEED,BSM_NAME=$BSM_NAME,BSM_FILE=$BSM_FILE,SM_FILE=$SM_FILE,SM_NAME=$SM_NAME,OUTFILE=$OUTFILE1,N=$N submit_run_trials.sbatch"
    echo $CMD
done
```
