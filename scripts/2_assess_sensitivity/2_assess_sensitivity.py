import numpy as np
import h5py as h5
import os

from scipy.optimize import ridder
from scipy.interpolate import interp1d
from typing import List

def initialize_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "--infile",
        type=str,
        required=True
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.95
    )
    args = parser.parse_args()
    return args

def decide_group_name(h5f: h5.File, basegroupname="results"):
    if basegroupname not in h5f.keys():
        return basegroupname
    idx = 1
    while True:
        groupname = f"{basegroupname}_{idx}"
        if groupname in h5f.keys():
            idx +=1
            continue
        break
    return groupname

def find_all_masses(h5f: h5.File) -> np.ndarray:
    ms = []
    for k, v in h5f.items():
        m = find_mass(v)
        if m in ms:
            continue
        ms.append(m)
    ms = np.array(ms)
    ms = ms[np.argsort(ms)]
    return ms

def find_relevant_keys(h5f: h5.File, m: float) -> List[str]:
    keys = []
    for k, v in h5f.items():
        if m!=find_mass(v):
            continue
        keys.append(k)
    return keys

def find_mass(gr: h5.Group) -> float:
    bsm_desc = gr.attrs["bsm_name"]
    m = float(bsm_desc.split("-")[2].replace("d", ".").replace("MeV", ""))
    return m

def find_coupling(gr: h5.Group) -> float:
    """
    Find the coupling from the metadata of a group

    params
    ______
    gr: the Group

    returns
    _______
    g: the coupling of the BSM model in that group
    """
    bsm_desc = gr.attrs["bsm_name"]
    logcoupling = float(
        bsm_desc.split("-")[3].replace("g", "").replace("dot", ".").replace("n", "-").replace("d", ".")
    )
    return 10 ** logcoupling

def find_q0(g: h5.Group, q: float) -> np.ndarray:
    """
    Find the quantile at which the signal-plus-background 
    trials cross an input quantile of the background-only trials

    params
    ______
    g: a Group containing signal and background results datasets
    q: quantile of background-only trials you want to find signal
        crossing for

    returns
    _______
    q0: the quantile at which the signal-plus-background trials are above
        the background-only trials
    """
    x = g.attrs["bsm_name"]
    bg90 = np.quantile(g["background"]["ts"][:], q)
    f = lambda x: np.quantile(g["signal"]["ts"][:], x) - bg90
    try:
        q0 = ridder(f, 0, 1)
    except ValueError as e:
        if repr(e)!="ValueError('f(a) and f(b) must have different signs')":
            raise e
        q0 = 0.0
    return q0

def main(args=None) -> None:
    if args is None:
        args = initialize_args()

    if not os.path.exists(args.outfile):
        with h5.File(args.outfile, "w") as _:
            pass

    with h5.File(args.infile) as h5f:
        ms = find_all_masses(h5f)
    
    exclusions = []
    sensitivities = []
    for m in ms:
        qs = []
        q0s = []
        couplings = []
        with h5.File(args.infile) as h5f:
            keys = find_relevant_keys(h5f, m)
        for key in keys:
            with h5.File(args.infile) as h5f:
                gr = h5f[key]
                q0 = find_q0(gr, args.quantile)
                coupling = find_coupling(gr)
                q0s = np.append(q0s, q0)
                qs = np.append(qs, np.quantile(gr["signal/ts"][:], 1-args.quantile))
                couplings = np.append(couplings, coupling)
        sorter = np.argsort(couplings)
        q0s = q0s[sorter]
        qs = qs[sorter]
        couplings = couplings[sorter]
        exclusion = couplings[qs > 0][0]
        interp = interp1d(np.log(couplings), q0s)
        f = lambda lg: interp(lg) - 0.5
        gsens = np.exp(ridder(f, np.log(couplings).min(), np.log(couplings).max()))
        sensitivities.append(gsens)
        exclusions.append(exclusion)

    with h5.File(args.outfile, "r+") as h5f:
        gn = decide_group_name(h5f)
        h5f.create_group(gn)
        h5f[gn].create_dataset("masses", data=ms)
        h5f[gn].create_dataset("sensitivities", data=sensitivities)
        h5f[gn].create_dataset("exclusions", data=exclusions)
        for k, v in vars(args).items():
            h5f[gn].attrs[k] = v

if __name__=="__main__":
    main()
