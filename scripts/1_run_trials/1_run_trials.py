import numpy as np
import h5py as h5

from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
from tqdm import tqdm

from sne_bsm import units, deserialize
from sne_bsm.likelihood import sig_likelihood, bg_likelihood
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
        "--seed",
        type=int,
        required=True
    )
    parser.add_argument(
        "--mismodeling_coefficient",
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
    parser.add_argument(
        "--hit_scaling",
        type=float,
        default=1.0
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
    for idx in itr:

        underlying_truth = signal_norm * nominal_bsm + mismodeling_coefficient * nominal_sm + nominal_bg
        data = np.random.poisson(underlying_truth)
        
        ts = -1
        ntries = 0

        while True:

            f = lambda x: np.sum(sig_likelihood([x[0], x[1]], data, nominal_bsm, nominal_sm, nominal_bg, sm_uncertainty))
            g = lambda x: np.sum(bg_likelihood(x, data, nominal_sm, nominal_bg, sm_uncertainty))

            x0g = [1 + (np.random.rand() * sm_uncertainty * 2 - sm_uncertainty)]
            resg = minimize(g, x0g, bounds=[(0.1, 5)], tol=1e-30)

            x0f = [0.5 + 0.3 * np.random.rand(), resg.x[0]]
            if ntries > 0:
                x0f[0] = np.random.uniform(0, 0.1)
            resf = minimize(f, x0f, bounds=[(0, 5), (0.1, 5)], tol=1e-30)

            ntries += 1

            ts = -2 * (f(resf.x) - g(resg.x))

            if ts >=0 or resf.x[0]==0:
                resfx = resf.x
                break
            elif abs(ts) < 1e-5 and f([0, resg.x[0]]) < f(resf.x):
                resfx = [0, resg.x[0]]
                break
            
            if ntries > 1000:
                raise RuntimeError("Too many tries")

        result = TrialsResults(resg.x[0], resfx[1], resfx[0], ts)
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

def main(args=None):
    if args is None:
        args = initialize_args()

    np.random.seed(args.seed)

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
    sm_hits *= args.hit_scaling
    
    # Compute hits from BSM model
    with h5.File(args.bsm_file, "r") as h5f:
        flux = deserialize(h5f[args.bsm_name])
    bsm_t, bsm_hits = flux.get_hits(
        model_file=args.tmpfile,
        tmax=100 * units["second"],
        dt=0.01*units["second"]
    )
    bsm_hits *= args.hit_scaling
    
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
    bg_hits *= args.hit_scaling
        
    # Run the trials for backgroun-only hypothesis
    bg_trials = run_background_trials(
        args.n,
        bsm_hits,
        sm_hits,
        bg_hits,
        args.sm_uncertainty,
        track=not args.no_track,
        mismodeling_coefficient=args.mismodeling_coefficient
    )
    # Run the trials for signal-plus-background hypothesis
    sig_trials = run_signal_trials(
        args.n,
        bsm_hits,
        sm_hits,
        bg_hits,
        args.sm_uncertainty,
        track=not args.no_track,
        mismodeling_coefficient=args.mismodeling_coefficient
    )
    save_trials_results(args.outfile, sig_trials, bg_trials, metadata=vars(args))

if __name__=="__main__":
    main()
