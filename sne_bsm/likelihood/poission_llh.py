import numpy as np

def poisson_loglikelihood(data: np.ndarray, model: np.ndarray) -> np.ndarray:
    """
    Equivalent log-likelihood from PDG statistics section. This has the advantage that
    one never needs to deal with the factorial or gamma functions.
    ref: https://pdg.lbl.gov/2020/reviews/rpp2020-rev-statistics.pdf, eq 40.16

    params
    ______
    data: experimental data array
    model: mean expected values of model to test

    returns
    _______
    out: log-likelihood in each bin

    raises
    ______
    ValueError: If data and model do not have same shape
    ValueError: If model is 0 in place where data is non-zero since this
        results in an infinite likelihood and should be resolved exernally
        to the function
    """
    
    if data.shape!=model.shape:
        raise ValueError("Incompatible data and model shapes")
    nonzero_mask = data > 0
    if np.any(model[nonzero_mask]==0):
        raise ValueError("Model is zero with non-zero data. LLH infinite")

    m = data > 0
    out = np.empty(data.shape)
    out[m] = -(model[m] - data[m] + data[m] * np.log(data[m] / model[m]))
    out[~m] = -(model[~m] - data[~m])

    return out

def sig_likelihood(
    x: np.ndarray,
    data: np.ndarray,
    nominal_bsm: np.ndarray,
    nominal_sm: np.ndarray,
    nominal_bg: np.ndarray,
    sm_uncertainty
):
    model = x[0] * nominal_bsm + x[1] * nominal_sm + nominal_bg
    nllh = -poisson_loglikelihood(data, model)
    #nllh += ((x[1] - 1) / sm_uncertainty) ** 2
    return nllh

def bg_likelihood(x, data, nominal_sm, nominal_bg, sm_uncertainty, sig_norm=0.0):
    nominal_bsm = np.zeros(data.shape)
    nllh = sig_likelihood(
        [sig_norm, x],
        data,
        nominal_bsm,
        nominal_sm,
        nominal_bg,
        sm_uncertainty=sm_uncertainty
    )
    return nllh
