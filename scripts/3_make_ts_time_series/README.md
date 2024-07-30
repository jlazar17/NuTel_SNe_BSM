```bash
OUTFILE=../../resources/ts_time_series.h5
for BSMNAME in dphi-dEdt-0d1MeV-gn9dot25-100s-nf_0 dphi-dEdt-150MeV-gn11dot8-100s-nf_0
do 
    python 3_make_ts_time_series.py \
        --bsm_file /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jlazar/SNe_BSM//serialized_majoran_fluxes.h5 \
        --sm_file /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jlazar/SNe_BSM//serialized_sm_flux.h5 \
        --bsm_name ${BSMNAME} \
        --sm_name sm_flux_0 \
        --seed 43 \
        --outfile ${OUTFILE} \
        -n 30000
done
```

```bash
INFILE=$OUTFILE
python plot_output.py \
    --infile ${INFILE} \
    --outfile ../../figures/ts_time_series.pdf \
    --trials_file /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jlazar/SNe_BSM/majoron_trials.h5
```
