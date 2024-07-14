# Nominal analysis

## Update and source `env.sh`

There are a few environmental variables that will be used to steer the scripts. Once you have edited them, you should run

```bash
source env.sh
```

## Download all the data

These are all the fluxes that we use in our analysis

```bash
TODO: put a download link
mv FILE $DATADIR
WD=$PWD
cd $DATADIR
unzip FILE
cd $WD
WD=
```

## Parameterize the tabulated fluxes

```bash
TODO
```

## Run trials

Here is some example usage for running the analysis on a slurm cluster. For this to work, you will need to edit the `call_run_trials.sbatch` in `../submit` to reflect the server you are working on

```bash
export OUTFILE1=${DATADIR}/majoron_trials.h5
export SEED=1
export BSM_FILE=${DATADIR}/serialized_majoran_fluxes.h5
export SM_FILE=${DATADIR}/serialized_sm_flux.h5
export SM_NAME=sm_flux_0
export N=50000

for A in `ls ${DATADIR}*.npy | xargs -n 1 basename`
do 
    L=${#A}
    BSM_NAME=${A:0:$((L - 4))}_0
    CMD="sbatch -D $PWD --export=SEED=$SEED,BSM_NAME=$BSM_NAME,BSM_FILE=$BSM_FILE,SM_FILE=$SM_FILE,SM_NAME=$SM_NAME,OUTFILE=$OUTFILE1,N=$N ../submit/call_run_trials.sbatch"
    $CMD
    SEED=$(($SEED+1))
done
```

## Assess the sensitivity

```bash
INFILE=$OUTFILE1
OUTFILE=${DATADIR}/majoron_sensitivities.h5
QUANTILE=0.95

python 2_assess_sensitivity.py --infile $INFILE --outfile $OUTFILE --quantile $QUANTILE
```

And voila you have sensitivities

# Mismodeling assessment

For this, we only need to rerun the last two steps.

## Run the trials

```bash
export MISMODELING_COEFFICIENT=1.2
export OUTFILE1=${DATADIR}/majoron_trials_mismodeling_${MISMODELING_COEFFICIENT}.h5
export BSM_FILE=${DATADIR}/serialized_majoran_fluxes.h5
export SM_FILE=${DATADIR}/serialized_sm_flux.h5
export SM_NAME=sm_flux_0
export N=50000
export SEED=1211

for A in `ls ${DATADIR}*.npy | xargs -n 1 basename`
do 
    L=${#A}
    BSM_NAME=${A:0:$((L - 4))}_0
    CMD="sbatch -D $PWD --export=SEED=$SEED,BSM_NAME=$BSM_NAME,BSM_FILE=$BSM_FILE,SM_FILE=$SM_FILE,SM_NAME=$SM_NAME,OUTFILE=$OUTFILE1,N=$N,MISMODELING_COEFFICIENT=$MISMODELING_COEFFICIENT ../submit/call_run_trials.sbatch"
    $CMD
    SEED=$(($SEED+1))
done
```
