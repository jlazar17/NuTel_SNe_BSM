import numpy as np
import h5py as h5
import os

from scipy.optimize import minimize
from tqdm import tqdm

from sne_bsm import units, deserialize
from sne_bsm.likelihood import sig_likelihood, bg_likelihood
from utils import make_groupname

def initialize_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "-n",
        type=int,
        required=True
    )
    parser.add_argument(
        "--bsm_file",
        type=str,
        required=True
    )
    parser.add_argument(
        "--sm_file",
        type=str,
        required=True
    )
    parser.add_argument(
        "--bsm_name",
        type=str,
        required=True
    )
    parser.add_argument(
        "--sm_name",
        type=str,
        required=True
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True
    )
    parser.add_argument(
        "--sm_uncertainty",
        type=float,
        default=0.2
    )
    parser.add_argument(
        "--no_track",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--tmpfile",
        type=str,
        default=".run_trials.txt"
    )
    args = parser.parse_args()
    return args


def compute_ts(
    n: int,
    nominal_bsm: np.ndarray,
    nominal_sm: np.ndarray,
    nominal_bg: np.ndarray,
    sm_uncertainty: float,
    track=True,
):
    res = np.empty(nominal_bsm.shape + (n,))

    underlying_truth = nominal_bg + nominal_sm + nominal_bsm
    itr = range(n)
    if track:
        itr = tqdm(itr)
    for idx in itr:
        data = np.random.poisson(underlying_truth)
        ntries = 0
        while True:
            ntries += 1
            f = lambda x: np.sum(
                sig_likelihood([x[0], x[1]], data, nominal_bsm, nominal_sm, nominal_bg, sm_uncertainty)
            )
            g = lambda x: np.sum(
                bg_likelihood(x, data, nominal_sm, nominal_bg, sm_uncertainty)
            )

            x0g = [1 + (np.random.rand() * sm_uncertainty * 2 - sm_uncertainty)]
            resg = minimize(g, x0g, bounds=[(0.1, 5)], tol=1e-30)

            x0f = [0.5 + 0.3 * np.random.rand(), resg.x[0]]
            if ntries > 0:
                x0f[0] = np.random.uniform(0, 0.1)
            resf = minimize(f, x0f, bounds=[(0, 5), (0.1, 5)], tol=1e-30)


            ts = -2 * (f(resf.x) - g(resg.x))

            if ts >=0 or resf.x[0]==0:
                resfx = resf.x
                break
            elif abs(ts) < 1e-5 and f([0, resg.x[0]]) < f(resf.x):
                resfx = [0, resg.x[0]]
                break
        a = sig_likelihood(resfx, data, nominal_bsm, nominal_sm, nominal_bg, sm_uncertainty)
        b = bg_likelihood(resg.x, data, nominal_sm, nominal_bg, sm_uncertainty)
        res[:, idx] = 2 * (b - a)
    return res

def main(args=None):
    if args is None:
        args = initialize_args()

    np.random.seed(args.seed)

    if not os.path.exists(args.outfile):
        with h5.File(args.outfile, "w") as _:
            pass


    # Compute hits from SM flux
    with h5.File(args.sm_file, "r") as h5f:
        sm_flux = deserialize(h5f[args.sm_name])
    sm_t, sm_hits = sm_flux.get_hits(
        tmax=100 * units["second"],
        model_file=args.tmpfile,
        dt=0.01*units["second"]
    )
    
    # Compute hits from BSM model
    with h5.File(args.bsm_file, "r") as h5f:
        flux = deserialize(h5f[args.bsm_name])
    bsm_t, bsm_hits = flux.get_hits(
        model_file=args.tmpfile,
        tmax=100 * units["second"],
        dt=0.01*units["second"]
    )
    
    # Make sure times are aligned for SM and BSM
    if np.any(sm_t!=bsm_t):
        raise ValueError("Hit times are different !")

    # Compute background hits
    bg_hits = flux.get_background(
        shape=bsm_hits.shape,
        model_file=args.tmpfile,
        tmax=100 * units["second"],
        dt=0.01*units["second"]
    )

    test_statistics = compute_ts(
        args.n,
        bsm_hits,
        sm_hits,
        bg_hits,
        args.sm_uncertainty,
        track = not args.no_track
    )

    with h5.File(args.outfile, "r+") as h5f:
        gn = make_groupname(h5f, "results")
        h5f[gn].create_dataset("test_statistic_series", data=test_statistics)
        h5f[gn].create_dataset("sm_hits", data=sm_hits)
        h5f[gn].create_dataset("bsm_hits", data=bsm_hits)
        h5f[gn].create_dataset("bg_hits", data=bg_hits)
        h5f[gn].create_dataset("times", data=bsm_t / units["second"])
        for k, v in vars(args).items():
            h5f[gn].attrs[k] = v

if __name__=="__main__":
    main()
