seed=987
for file in `ls /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jlazar/SNe_BSM/*.npy | xargs -n 1 basename`
do 
    l=${#file}
    bsmname=${file:0:$(($l-4))}_0
    python c0_run_trials.py \
        --bsm_file /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jlazar/SNe_BSM/serialized_majoron_fluxes.h5 \
        --real_sm_file /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jlazar/SNe_BSM/serialized_sm_27solmass.h5 \
        --fake_sm_file /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jlazar/SNe_BSM/serialized_sm_flux.h5 \
        --bsm_name ${bsmname} \
        --real_sm_name fluxactivenuEBar-27PNS_0 \
        --fake_sm_name sm_flux_0 \
        --seed ${seed} \
        --outfile ./test.h5 \
        -n 10000
    seed=$(($seed+1))
done
