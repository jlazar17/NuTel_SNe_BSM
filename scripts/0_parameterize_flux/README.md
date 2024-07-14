# Parameterize fluxes

The scripts in this directory will take tabulated fluxes and convert them to the parameterization expected by `SNEWPY`.
This assumes that you have sourced `../env.sh` and as such `DATADIR` is defined.
Additionally, it assumes that all the fluxes you want to parametrize as tabulated per flavor in `.npy` files in `DATADIR`.

## Running the Python script

The infile should be a numpy file with per-flavor fluxes.
The outfile will be a h5 file with a group that shares a name with the infile, minus the extension.
The thinning parameter takes every `thin` time step.
This can speed things up if the have fluxes that were computed with a time-step that is more fine than is important for your application.

```bash
THIN=10
INFILE=...
OUTFILE=${DATADIR}/serialized_majoron_fluxes.h5
python 0_parameterize_flux.py --infile ${INFILE} --outfile ${OUTFILE} --thin ${THIN}
```

## Using the slurm submission script

If you happen to be using a cluster with SLURM scheduling, you can use `submit_parameterize_flux.sbatch` as an example of how to do submission.
You will have to change the error and output directories in that script to places that you have write permission, and will have to change the requested partitions.

```bash
OUTFILE=${DATADIR}/serialized_majoron_fluxes.h5
for INFILE in `ls $DATADIR/*.npy`
do
    CMD="sbatch -D $PWD --export=INFILE=$INFILE,OUTFILE=$OUTFILE,THIN=10 submit_parameterize_flux.sbatch"
    $CMD
done
```
