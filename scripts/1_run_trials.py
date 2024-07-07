import numpy as np
import h5py as h5

from dataclasses import dataclass
from scipy.optimize import minimize
from tqdm import tqdm

from sne_bsm import units, deserialize
from likelihood import sig_likelihood, bg_likelihood
from utils import save_trials_results

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
        "--midmodeling_coefficient",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True
    )
    parser.add_argument(
        "--asteria_path",
        type=str,
        default=""
    )
    parser.add_argument(
        "--sm_uncertainty",
        type=float,
        default=0.2
    )
    args = parser.parse_args()
    return args

@dataclass
class TrialsResults:
    bg_sm_norm: float
    sb_sm_norm: float
    sb_bsm_norm: float
    ts: float

def run_trials(
    ntrial: int,
    nominal_bsm: np.ndarray,
    nominal_sm: np.ndarray,
    nominal_bg: np.ndarray,
    sm_uncertainty: float,
    signal_norm: float,
    track=True,
    mismodeling_coefficient=1.0
):
    
    results = []
    itr = range(ntrial)
    if track:
        itr = tqdm(itr)
    for _ in itr:
        data = np.random.poisson(signal_norm * nominal_bsm + mismodeling_coefficient * nominal_sm + nominal_bg)
        
        f = lambda x: np.sum(sig_likelihood([x[0], x[1]], data, nominal_bsm, nominal_sm, nominal_bg, sm_uncertainty))
        g = lambda x: np.sum(bg_likelihood(x, data, nominal_sm, nominal_bg, sm_uncertainty))

        x0 = [np.log(1), 1 + (np.random.rand() * sm_uncertainty * 2 - sm_uncertainty)]

        resf = minimize(f, x0, bounds=[(0, 20), (0, 5)])
        resg = minimize(g, x0[1], bounds=[(0, 5)])
        
        ts = -2 * (f(resf.x) - g(resg.x))
        result = TrialsResults(resg.x[0], resf.x[1], resf.x[0], ts)
        results.append(result)
    return results

def run_background_trials(
    ntrial: int,
    nominal_bsm: np.ndarray,
    nominal_sm: np.ndarray,
    nominal_bg: np.ndarray,
    sm_uncertainty: float,
    track=True,
    mismodeling_coefficient=1.0

):  
    results = run_trials(
        ntrial,
        nominal_bsm,
        nominal_sm,
        nominal_bg,
        sm_uncertainty,
        0.0,
        track=track,
        mismodeling_coefficient=mismodeling_coefficient
    )
    return results

def run_signal_trials(
    ntrial: int,
    nominal_bsm: np.ndarray,
    nominal_sm: np.ndarray,
    nominal_bg: np.ndarray,
    sm_uncertainty: float,
    track=True,
    mismodeling_coefficient=1.0
):
    
    results = run_trials(
        ntrial,
        nominal_bsm,
        nominal_sm,
        nominal_bg,
        sm_uncertainty,
        1.0,
        track=track,
        mismodeling_coefficient=mismodeling_coefficient
    )
    return results

def main():
    args = initialize_args()

    if args.asteria_path:
        import sys
        os.environ["ASTERIA"] = args.asteria_path
        sys.path.append(f"{args.asteria_path}/python/")

    # Compute hits from SM flux
    with h5.File(args.sm_file, "r") as h5f:
        sm_flux = deserialize(h5f[args.sm_name])
    sm_t, sm_hits = sm_flux.get_hits(
        tmax=100 * units["second"],
        model_file="magnetic_moment.txt",
        dt=0.01*units["second"]
    )
    
    # Compute hits from BSM model
    with h5.File(args.bsm_file, "r") as h5f:
        flux = deserialize(h5f[args.bsm_name])
    bsm_t, bsm_hits = flux.get_hits(
        model_file="magnetic_moment.txt",
        tmax=100 * units["second"],
        dt=0.01*units["second"]
    )
    
    # Make sure times are aligned for SM and BSM
    if np.any(sm_t!=bsm_t):
        raise ValueError("Hit times are different !")

    # Compute background hits
    bg_hits = flux.get_background(
        shape=bsm_hits.shape,
        model_file="magnetic_moment.txt",
        tmax=100 * units["second"],
        dt=0.01*units["second"]
    )
        
    # Run the trials for backgroun-only hypothesis
    bg_trials = run_background_trials(args.n, bsm_hits, sm_hits, bg_hits, args.sm_uncertainty)
    # Run the trials for signal-plus-background hypothesis
    sig_trials = run_signal_trials(args.n, bsm_hits, sm_hits, bg_hits, args.sm_uncertainty)
    save_trials_results(args.outfile, sig_trials, bg_trials, metadata=vars(args))

if __name__=="__main__":
    main()
