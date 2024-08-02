seed=987
for bsmname in `cat bsmnames.txt`
do
    python c1_run_trials.py \
        --bsm_file /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jlazar/SNe_BSM/serialized_majoron_fluxes.h5 \
        --fake_sm_file /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jlazar/SNe_BSM/serialized_sm_27solmass.h5 \
        --real_sm_file /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jlazar/SNe_BSM/serialized_sm_flux.h5 \
        --bsm_name ${bsmname} \
        --fake_sm_name fluxactivenuEBar-27PNS_0 \
        --real_sm_name sm_flux_0 \
        --seed ${seed} \
        --outfile ./test.h5 \
        -n 1000
    seed=$(($seed+1))
done
