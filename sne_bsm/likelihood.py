import numpy as np

from .units import units

TMAX = 25 * units["second"]
TMIN = 0 * units["second"]
TMINS = np.linspace(TMIN, TMAX, 1001)
DTS = np.linspace(0.0, 25, 250) * units["second"]

def find_significance_matrix(sig_hits, sm_hits, bg_hits, times, tmins=None, dts=None):
    if tmins is None:
        tmins = TMINS
    if dts is None:
        dts = DTS
    significance = np.full(tmins.shape + dts.shape, np.nan)

    for idx, tmin in enumerate(tmins):
        for jdx, dt in enumerate(dts):
            m = np.logical_and(tmin < times, times < tmin+dt)
            if not m.sum():
                continue
            significance[idx, jdx] = likelihood(sig_hits[m], sm_hits[m], bg_hits[m])
    return significance, tmins, dts

def likelihood(sig_hits, sm_hits, bg_hits):
    n_obs = (bg_hits + sm_hits).sum()
    n_exp = (sig_hits + bg_hits + sm_hits).sum()
    llh = 2 * (n_exp - n_obs)
    if n_obs > 0:
        llh += 2 * n_obs * np.log(n_obs / n_exp)
    return llh