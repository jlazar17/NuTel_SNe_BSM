# Asses the sensitivity of the model

In this step, we will use the file produced in the last step to evaluate different metrics of our sensitivity to a model.

First, we point to the infile from the last step:

```bash
infile=${DATADIR}/trials.h5
```

We will then define an outfile and optionally the name of the group in the output HDF5 file.
By default this will be stored in `results`.
If this is the desired behavior, then the corresponding CL argument does not need to be specifified.

```bash
outfile=${DATADIR}/results.h5
outgroup=results
```

We can then run the script with:

```bash
python 2_assess_sensitivity.py --infile $infile --outfile $outfile --outgroup $outgroup
```

Each group in the file will have three datasets named `masses`, `sensitivities`, and `exclusions`.
The `sensitivities` dataset has the coupling at which 50% of the signal-plus-background trials exceeds 95% of the background-only trials.
The `exclusions` dataset has the coupling when 95% of the signal-plus-background trials are above 0.0.
Thus, these fields tell you the coupling at which one would observe a model at 95% 50% of the time and the coupling which one would exclude at 95% in the event of a null observation, respectively.
One can change the quantile in question from 95% to any other quantile by supplying the `--quantile q0` command line argument, where `0 < q0 < 1`.
