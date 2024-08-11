import numpy as np
import h5py as h5

from dataclasses import dataclass
from scipy.optimize import minimize

from sne_bsm import units, deserialize
from sne_bsm.likelihood import sig_likelihood, bg_likelihood
from utils import save_trials_results, get_snewpy_hits

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
        "--fake_sm_file",
        type=str,
        required=True
    )
    parser.add_argument(
        "--bsm_name",
        type=str,
        required=True
    )
    parser.add_argument(
        "--fake_sm_name",
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
    args = parser.parse_args()
    return args

@dataclass
class TrialsResults:
    llh_bg: float
    llh_sb: float
    bg_norm_bg: float
    bg_norm_sb: float
    sig_norm_sb: float

def run_trials(
    ntrial: int,
    nominal_bsm: np.ndarray,
    nominal_real_sm: np.ndarray,
    nominal_fake_sm: np.ndarray,
    nominal_bg: np.ndarray,
    sm_uncertainty: float,
    track=True,
    mismodeling_coefficient=1.0,
    seed=None
):
    real_results, fake_results = [], []
    itr = range(ntrial)
    if track:
        try:
            from tqdm import tqdm
            itr = tqdm(itr)
        except ImportError:
            pass

    if seed is not None:
        np.random.seed(seed)
    underlying_truth = nominal_real_sm + nominal_bg
    for idx in itr:

        data = np.random.poisson(underlying_truth)

        f_real = lambda x: np.sum(sig_likelihood(x, data, nominal_bsm, nominal_real_sm, nominal_bg, sm_uncertainty))
        f_fake = lambda x: np.sum(sig_likelihood(x, data, nominal_bsm, nominal_fake_sm, nominal_bg, sm_uncertainty))
        g_real = lambda x: np.sum(bg_likelihood(x, data, nominal_real_sm, nominal_bg, sm_uncertainty))
        g_fake = lambda x: np.sum(bg_likelihood(x, data, nominal_fake_sm, nominal_bg, sm_uncertainty))

        xf_real = [0.5 + np.random.rand(), 1 + (np.random.rand() * sm_uncertainty * 2 - sm_uncertainty)]
        xf_fake = [0.5 + np.random.rand(), 1 + (np.random.rand() * sm_uncertainty * 2 - sm_uncertainty)]
        xg_real = [1 + (np.random.rand() * sm_uncertainty * 2 - sm_uncertainty)]
        xg_fake = [1 + (np.random.rand() * sm_uncertainty * 2 - sm_uncertainty)]

        resf_real = minimize(f_real, xf_real, bounds=[(0.0, 5), (0.0, 20)])
        resf_fake = minimize(f_fake, xf_fake, bounds=[(0.0, 5), (0.0, 20)])
        resg_real = minimize(g_real, xg_real, bounds=[(0.0, 20)])
        resg_fake = minimize(g_fake, xg_fake, bounds=[(0.0, 20)])

        res_real = TrialsResults(
            g_real(resg_real.x),
            f_real(resf_real.x),
            resg_real.x,
            resf_real.x[1],
            resf_real.x[0]
        )

        res_fake = TrialsResults(
            g_fake(resg_fake.x),
            f_fake(resf_fake.x),
            resg_fake.x,
            resf_fake.x[1],
            resf_fake.x[0]
        )

        real_results.append(res_real)
        fake_results.append(res_fake)

    return real_results, fake_results

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

    real_sm_t, real_sm_hits = get_snewpy_hits()

    # Compute hits from SM flux
    with h5.File(args.fake_sm_file, "r") as h5f:
        fake_sm_flux = deserialize(h5f[args.fake_sm_name])

    fake_sm_t, fake_sm_hits = fake_sm_flux.get_hits(
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
    if np.any(fake_sm_t!=bsm_t):
        raise ValueError("Hit times are different !")
    if np.any(real_sm_t!=bsm_t):
        raise ValueError("Hit times are different !")

    # Compute background hits
    bg_hits = flux.get_background(
        shape=bsm_hits.shape,
        model_file=args.tmpfile,
        tmax=100 * units["second"],
        dt=0.01*units["second"]
    )

    real_res, fake_res = run_trials(
        args.n,
        bsm_hits,
        real_sm_hits,
        fake_sm_hits,
        bg_hits,
        args.sm_uncertainty,
        track=not args.no_track,
        mismodeling_coefficient=args.mismodeling_coefficient,
        seed=args.seed
    )
    save_trials_results(args.outfile, real_res, fake_res, metadata=vars(args))

if __name__=="__main__":
    main()
