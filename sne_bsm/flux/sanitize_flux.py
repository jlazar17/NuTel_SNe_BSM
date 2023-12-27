import numpy as np

def find_first_spike(arr: np.ndarray, thresh=1e-10):

    if len(arr)==0:
        return 0
    
    if arr[0]==0 or arr[-1]==0:
        raise ValueError("Array cannot start with zero")
    
    for idx, (v1, v2) in enumerate(zip(arr, arr[1:])):
        if v2==0:
            continue
        if v1 / v2 < thresh:
            return idx + 1
    return 0

def argtrim_zeros(arr: np.ndarray):
    l, r = 0, len(arr)
    
    # Catch length 0 arrays
    if r==0:
        return l, r
    
    while arr[l]==0:
        l += 1
        # Catch arrays that are full 0s
        if l==len(arr):
            return l, r
    
    while arr[r-1]==0:
        r -= 1
        
    return l, r

def sanitize_flux(energies: np.ndarray, flux: np.ndarray):
    l, r = argtrim_zeros(flux)
    flux = flux[l:r]
    energies = energies[l:r]

    idx = find_first_spike(flux)
    flux = flux[idx:]
    energies = energies[idx:]
    return energies, flux
