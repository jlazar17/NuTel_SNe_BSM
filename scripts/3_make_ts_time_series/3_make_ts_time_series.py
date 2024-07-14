import numpy as np
import h5py as h5

from scipy.optimize import minimize
from tqdm import tqdm

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
    bsm_hits: np.ndarray,
    sm_hits: np.ndarray,
    bg_hits: np.ndarray,
    sm_uncertainty: float,
    track=True,
):
    res = np.empty(bsm_hits.shape + (n,))

    underlying_truth = bg_hits + sm_hits + bsm_hits
    itr = range(n)
    if track:
        itr = tqdm(itr)
    for idx in itr:
        data = np.random.poisson(underlying_truth)
        f = lambda x: np.sum(sig_likelihood([x[0], x[1]], data, bsm_hits, sm_hits, bg_hits, sm_uncertainty))
        g = lambda x: np.sum(bg_likelihood(x, data, sm_hits, bg_hits, sm_uncertainty))
        x0 = [0.5 + np.random.rand(), 1 + (np.random.rand() * sm_uncertainty * 2 - sm_uncertainty)]
        resf = minimize(f, x0, bounds=[(0, 20), (0.1, 5)])
        resg = minimize(g, x0[1], bounds=[(0.1, 5)])
        a = sig_likelihood(resf.x, data, bsm_hits, sm_hits, bg_hits, sm_uncertainty)
        b = bg_likelihood(resg.x, data, sm_hits, bg_hits, sm_uncertainty)
        res[:, idx] = 2 * (b - a)
    return res

def main(args=None):
    if args is None:
        args = initialize_args()

    np.random.seed(args.seed)

    if not os.path.exists(args.outfile):
        with h5.File(args.outfile, "w") as _:
            pass

    if args.asteria_path:
        import sys
        os.environ["ASTERIA"] = args.asteria_path
        sys.path.append(f"{args.asteria_path}/python/")

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
        
