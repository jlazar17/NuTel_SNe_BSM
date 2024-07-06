import numpy as np
import h5py as h5

from sne_bsm import units
from sne_bsm.flux import compute_params, ParameterizedFlux


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
        "--outgroup",
        type=str,
        default="",
        help="group name in h5 file. If this is left blank, the \
        name of the infile will be used with extensions removed"
    )
    parser.add_argument(
        "--sn_distance",
        type=float,
        default=10,
        help="Distance to the SN in kiloparsecs."
    )
    parser.add_argument(
        "--is_total",
        action="store_true",
        default=False,
        help="If the flux in the infile is the total flux rather than the per-flavor \
        flux. If this flag is included, the total flux will be divided by 6 to ge the \
        per-flavor flux"
    )
    parser.add_argument(
        "--thin",
        type=int,
        default=1,
        help="Factor to reduce the number of times at which the flux is sampled."
    )
    parser.add_argument(
        "--no_track",
        action="store_true",
        default=False
    )
    args = parser.parse_args()
    return args

def load_infile(infile):
    try:
        # Check if it is a numpy file
        data = np.load(infile)
    except ValueError: # Catch if it is a text-based file
        data = np.genfromtxt(infile)
        if np.any(np.isnan(data)): # This means it is CSV
            data = np.genfromtxt(infile, delimiter=",")
    if np.any(np.isnan(data)):
        raise ValueError(f"Don't know how to load data from {args.infile}")
    return data

def main():
    args = initialize_args()
    data = load_infile(args.infile)
    times = np.sort(np.unique(data[:, 0])) * units["second"]
    energies = np.sort(np.unique(data[:, 1])) * units["MeV"]
    fluxes = np.empty(times.shape + energies.shape + (3,),)
    for idx in range(len(times)):
        flux = data[idx*len(energies):(idx+1)*len(energies), 2] / units["MeV"] / units["second"]
        flux = np.where(flux >= 0, flux, 0) # Sometimes numerical issues can sneak in to give negative fluxes
        fluxes[idx, :, 0] = flux # nue
        fluxes[idx, :, 1] = flux # nuebar
        fluxes[idx, :, 2] = flux # nux per flavor
    if args.is_total: # Convert to per flavor flux
        fluxes /= 6

    times, fluxes = times[::args.thin], fluxes[::args.thin, :, :]

    # Parametrize the tabulated data
    params = []
    itr = enumerate(times)
    if not args.no_track:
        from tqdm import tqdm
        itr = enumerate(tqdm(times))
    for idx, time in itr:
        nue_param = compute_params(energies, fluxes[idx, :, 0])
        nuebar_param = compute_params(energies, fluxes[idx, :, 1])
        nux_param = compute_params(energies, fluxes[idx, :, 2])
        params.append((time, nue_param, nuebar_param, nux_param))

    # Make the Flux object
    flux = ParameterizedFlux(params, args.sn_distance * units["kpc"])

    # Serialize the flux object
    outgroup = args.outgroup
    if not outgroup:
        outgroup = args.infile.split(".")[-2].split("/")[-1]
    print(outgroup)
    flux.serialize(args.outfile, groupname=outgroup)

if __name__=="__main__":
    main()
