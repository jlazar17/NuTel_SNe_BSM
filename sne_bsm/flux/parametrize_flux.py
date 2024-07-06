import numpy as np
import h5py as h5
import os

from tqdm import tqdm
from dataclasses import dataclass
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import gamma
from asteria.simulation import Simulation

from .flux import Flux
from .sanitize_flux import sanitize_flux
from ..units import units

from typing import Tuple, List

@dataclass
class Params:
    L: float
    e_exp: float
    rms: float
    alpha: float

    def __add__(self, other):
        if not(isinstance(self, Params) and isinstance(other, Params)):
            raise ValueError()
        return Params(
            self.L + other.L,
            self.e_exp + other.e_exp,
            self.rms + other.rms,
            self.alpha + other.alpha
        )

    def __sub__(self, other):
        if not(isinstance(self, Params) and isinstance(other, Params)):
            raise ValueError()
        return Params(
            self.L - other.L,
            self.e_exp - other.e_exp,
            self.rms - other.rms,
            self.alpha - other.alpha
        )

    def __rmul__(self, other):
        return Params(
            other * self.L,
            other * self.e_exp,
            other * self.rms,
            other * self.alpha,
        )

    def __mul__(self, other):
        return Params(
            other * self.L,
            other * self.e_exp,
            other * self.rms,
            other * self.alpha,
        )


def parameterized_flux_from_files(
    nuefile: str,
    nuebarfile: str,
    nuxfile: str,
    dist: float,
    thin=1,
    track=True
):
    nuearr = massage_infile(nuefile)
    nuebararr = massage_infile(nuebarfile)
    nuxarr = massage_infile(nuxfile)
    _validate_fluxes(nuearr, nuebararr, nuxarr)
    params = parameterize(nuearr, nuebararr, nuxarr, thin=thin, track=track)
    return ParameterizedFlux(params, dist)
 
def parameterized_flux_from_h5(
    group: h5.Group,
    dist: float,
    thin: int=1,
    track: bool=True
):
    times = group["times"][::thin]
    energies = group["energies"][:]
    fluxes = group["fluxes"][::thin, :, :]
    params = []
    itr = enumerate(times)
    if track:
        itr = enumerate(tqdm(times))
    for idx, time in itr:
        nue_param = compute_params(energies, fluxes[idx, :, 0])
        nuebar_param = compute_params(energies, fluxes[idx, :, 1])
        nux_param = compute_params(energies, fluxes[idx, :, 2])
        params.append((time, nue_param, nuebar_param, nux_param))
    return ParameterizedFlux(params, dist)


class ParameterizedFlux(Flux):

    def __init__(
        self,
        params: List[Params],
        dist: float,
        nuearr: np.ndarray=None,
        nuebararr: np.ndarray=None,
        nuxarr: np.ndarray=None,
    ):
        """
        Initialize parametrized flux

        params
        ______
        nuefile: Text file in which electron neutrino flux is stored
        nuebarfile: Text file in which electron antineutrino flux is stored
        nuxfile: Text file in which all other neutrino fluxes are stored
        dist: Distance at which the SN was simulated
        """
        self._params = params
        self._times = [param[0] for param in params]
        self._dist = dist
        #self._nuearr = nuearr
        #self._nuebararr = nuebararr
        #self._nuxarr = nuxarr

    @property
    def dist(self):
        return self._dist

    def __len__(self) -> int:
        return self._shape[0]

    @property
    def params(self) -> List[Tuple[float, Params, Params, Params]]:
        return self._params

    def get_background(
        self,
        tmin=-1.0*units["second"],
        tmax=20.0*units["second"],
        dt=0.001*units["second"],
        emin=0*units["MeV"],
        emax=400*units["MeV"],
        de=0.5*units["MeV"],
        mixing_scheme="AdiabaticMSW",
        mass_ordering="normal",
        model_file="parameterized_model.txt",
        force_write=False,
        keep_model_file=False,
        dist0=units["m"],
        shape=None
    ):
        from astropy import units as u
        sim = self.get_asteria_sim(
            tmin=tmin,
            tmax=tmax,
            dt=dt,
            emin=emin,
            emax=emax,
            de=de,
            mixing_scheme=mixing_scheme,
            mass_ordering=mass_ordering,
            model_file=model_file,
            force_write=force_write,
            keep_model_file=keep_model_file,
            dist0=dist0
        )
        # TODO this is dangerous. Figure out if this is right
        if shape is None:
            shape = int((tmax - tmin) / dt) + 1
        hits = sim.detector.i3_bg(dt / units["second"] * u.s, size=shape) + sim.detector.dc_bg(dt / units["second"] * u.s, size=shape)
        return hits

    def get_asteria_sim(
        self,
        tmin=-1.0*units["second"],
        tmax=20.0*units["second"],
        dt=0.001*units["second"],
        emin=0*units["MeV"],
        emax=400*units["MeV"],
        de=0.5*units["MeV"],
        mixing_scheme="AdiabaticMSW",
        mass_ordering="normal",
        model_file="parameterized_model.txt",
        force_write=False,
        keep_model_file=False,
        dist0=units["m"]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the number of hits in the detector as a function of time

        params
        ______
        [tmin]: Minimum time relative to tbounce. Default -1.0 s
        [tmax]: Maximum time relative to tbounce. Default 20.0 s
        [dt]: time step. Smaller will take longer. Default 0.01 s
        [emin]: Minimum energy to integrate over, Default 0 MeV
        [emax]: Maximum energy to integrate over. Default 400 MeV
        [de]: Energy step. Default 0.5 MeV. TODO Find out what this does.
        [mixing_scheme]:
        [mass_ordering]: Neutrino mass ordering. Must be in
            ["normal", "inverted"]. Default "normal"
        [model_file]: File to which to write the parameters. Default "parameterized_model.txt"
        [force_write]: Whether to overwrite `model_file` if it exists. Default False
        [keep_model_file]: Whether to delete the parameter file. Default False
        [dist0]: Distance from the detector to pass into SNEWPY. This is mostly irrelevant,
            but it should be a distance over which oscillations are negligible.

        returns
        _______
        t: Array of times
        hits: Hits in each time bin

        """
        import astropy.units as u
        self._write_params(
            outfile=model_file,
            scale=(dist0 / self.dist)**2
        )
        model = {
            'name': 'Analytic3Species',
            'param': {
                'filename': model_file
            }
        }
        sim_kwargs = {
            "distance": dist0 / units["m"] * u.m,
            "model": model,
            "Emin": emin / units["MeV"] * u.MeV,
            "Emax": emax / units["MeV"] * u.MeV,
            "dE": de / units["MeV"] * u.MeV,
            "tmin": tmin / units["second"] * u.s,
            "tmax": tmax / units["second"] * u.s,
            "dt": 1e-3 * u.s,
            #"dt": dt / units["second"] * u.s,
            "mixing_scheme": mixing_scheme,
            "hierarchy": mass_ordering,
        }
        sim = Simulation(**sim_kwargs)
        sim.run()
        if not keep_model_file:
            from os import remove
            remove(model_file)
        return sim

    def get_hits(
        self,
        tmin=-1.0*units["second"],
        tmax=20.0*units["second"],
        dt=0.001*units["second"],
        emin=0*units["MeV"],
        emax=400*units["MeV"],
        de=0.5*units["MeV"],
        mixing_scheme="AdiabaticMSW",
        mass_ordering="normal",
        model_file="parameterized_model.txt",
        force_write=False,
        keep_model_file=False,
        dist0=units["m"]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the number of hits in the detector as a function of time

        params
        ______
        [tmin]: Minimum time relative to tbounce. Default -1.0 s
        [tmax]: Maximum time relative to tbounce. Default 20.0 s
        [dt]: time step. Smaller will take longer. Default 0.01 s
        [emin]: Minimum energy to integrate over, Default 0 MeV
        [emax]: Maximum energy to integrate over. Default 400 MeV
        [de]: Energy step. Default 0.5 MeV. TODO Find out what this does.
        [mixing_scheme]:
        [mass_ordering]: Neutrino mass ordering. Must be in
            ["normal", "inverted"]. Default "normal"
        [model_file]: File to which to write the parameters. Default "parameterized_model.txt"
        [force_write]: Whether to overwrite `model_file` if it exists. Default False
        [keep_model_file]: Whether to delete the parameter file. Default False
        [dist0]: Distance from the detector to pass into SNEWPY. This is mostly irrelevant,
            but it should be a distance over which oscillations are negligible.

        returns
        _______
        t: Array of times
        hits: Hits in each time bin

        """
        from astropy import units as u
        sim = self.get_asteria_sim(
            tmin=tmin,
            tmax=tmax,
            dt=dt,
            emin=emin,
            emax=emax,
            de=de,
            mixing_scheme=mixing_scheme,
            mass_ordering=mass_ordering,
            model_file=model_file,
            force_write=force_write,
            keep_model_file=keep_model_file,
            dist0=dist0
        )
        if dt!=1e-3*units.second:
            sim.rebin_result(dt / units["second"] * u.s)
        t, hits = sim.detector_signal(dt / units["second"] * u.s)
        return t.value * units["second"], hits
        
    def _write_params(self, outfile: str, scale: float=1.0):
        params = np.empty((len(self.params), 13))
        for idx, p in enumerate(self.params):
            time, pnue, pnuebar, pnux = p
            # Energy unit conversion factor
            euc = 1 / units["erg"]
            #euc = units["MeV"] / units["erg"]
            # Factor for integrating over sphere
            sph = 4*np.pi * self._dist**2 / units["cm"]**2
            # Combine em
            prefactor = euc * sph * scale

            params[idx, 0] = time / units["second"]
            params[idx, 1] = pnue.L * prefactor * units.second
            params[idx, 2] = pnuebar.L * prefactor * units.second
            params[idx, 3] = pnux.L * prefactor * units.second
            params[idx, 4] = pnue.e_exp / units.MeV
            params[idx, 5] = pnuebar.e_exp / units.MeV
            params[idx, 6] = pnux.e_exp / units.MeV
            params[idx, 7] = pnue.rms / units.MeV
            params[idx, 8] = pnuebar.rms / units.MeV
            params[idx, 9] = pnux.rms / units.MeV
            params[idx, 10] = pnue.alpha
            params[idx, 11] = pnuebar.alpha
            params[idx, 12] = pnux.alpha
        with open(outfile, "w") as outf:
            outf.write("TIME L_NU_E L_NU_E_BAR L_NU_X E_NU_E E_NU_E_BAR E_NU_X RMS_NU_E RMS_NU_E_BAR RMS_NU_X ALPHA_NU_E ALPHA_NU_E_BAR ALPHA_NU_X\n")
            for param in params:
                outf.write(" ".join([str(x) for x in param]) + "\n")

    def get_flux(self, t, e, nutype):
        ps = self._lerp_params(t)
        p = ps[nutype]
        s = spectrum(p.L, p.e_exp, p.alpha)
        return s(e)


    def _lerp_params(self, t0):
        if t0 < self.params[0][0] or self.params[-1][0]< t0:
            raise ValueError("Time out of range")
            
        idx = 0
        time, p0, p1, p2 = self.params[idx]
        old_time, old_p0, old_p1, old_p2 = time, p0, p1, p2
        while time < t0:
            old_time, old_p0, old_p1, old_p2 = time, p0, p1, p2
            idx += 1
            time, p0, p1, p2 = self.params[idx]
        f = (t0 - old_time) / (time - old_time)
        new_p0 = old_p0 + f * (p0 - old_p0)
        new_p1 = old_p1 + f * (p1 - old_p1)
        new_p2 = old_p2 + f * (p2 - old_p2)
        return new_p0, new_p1, new_p2

    def serialize(
        self,
        filename: str,
        groupname:str ="parameterized_flux",
        **kwargs
    ):
        serialize(self, filename, groupname=groupname, **kwargs)

def old_compute_moment(arr: np.ndarray, k: int) -> float:
    """
    Compute the kth moment for an energy distribution
    """
    if arr.shape[1] !=3:
        raise ValueError("Invalid input data shape")
    if len(np.unique(arr[:, 0])) > 1:
        raise ValueError("Input array is for different times")
    # remove any leading / trailing zeros in the flux
    idx_min = 0
    #while arr[idx_min, 2]==0:
    for idx_min in range(arr.shape[0]):
        if arr[idx_min, 2] > 0:
            break
    # Early return if allentries are zero
    if idx_min+1==arr.shape[0]:
        return 0
    idx_max = arr.shape[0]
    while arr[idx_max-1, 2]==0:
        idx_max -= 1
    arr = arr[idx_min:idx_max]
    
    e = arr[:, 1]
    flux = arr[:, 2]
    interp = interp1d(e, np.power(e, k) * flux)
    hack = interp(e.min())
    f = lambda e: interp(e) / hack
    m, err = quad(f, e.min(), e.max())
    if m==0:
        return 0
    if err / m > 1e-2:
        print(flux)
        print(e)
        raise ValueError(f"Error {err / m} too big :-(")
    return m * hack

def compute_params(energies, flux, rtol=1e-7):

    energies, flux = sanitize_flux(energies, flux)

    if len(flux)<=1:
        return Params(0, 0, 0, 0)

    norm = compute_moment(energies, flux, 0)
    L = compute_moment(energies, flux, 1)
    e_exp = L / norm
    e2_exp = compute_moment(energies, flux, 2) / norm
    rms = np.sqrt(e2_exp)
    alpha = (2 * e_exp**2 - e2_exp) / (e2_exp - e_exp**2)
    p = (L, e_exp, rms, alpha)

    return Params(*p)

def old_compute_params(arr: np.ndarray):
    """
    Compute the parameters that SNEWPY understands from 
    a particular point in time.

    params
    ______
    arr:

    returns
    _______
    p: `Params` for for the flux at that point in time
    """
    # Carry out the computation
    norm = old_compute_moment(arr, 0)
    if norm==0:
        return Params(0, 0, 0, 0)
    L = old_compute_moment(arr, 1)
    e_exp = L / norm
    e2_exp = old_compute_moment(arr, 2) / norm
    rms = np.sqrt(e2_exp)
    alpha = (2 * e_exp**2 - e2_exp) / (e2_exp - e_exp**2)
    p = (L, e_exp, rms, alpha)
    if np.any(np.isnan(p)):
        p = (0, 0, 0, 0)
    return Params(*p)

def massage_infile(infile: str) -> np.ndarray:
    _, ext = os.path.splitext(infile)
    loadf = np.load
    # If it's not a numpy file, it has to be a csv
    if ext!=".npy":
        loadf = lambda f: np.genfromtxt(f, delimiter=",")
    a = loadf(infile)
    #a = np.genfromtxt(infile, delimiter=",")
    ntimes = len(np.unique(a[:, 0]))
    nenergies = int(a.shape[0] / ntimes)
    return a.reshape((ntimes, nenergies, 3))

def _validate_fluxes(nuearr, nuebararr, nuxarr):
    if not (nuearr.shape == nuebararr.shape == nuxarr.shape):
        raise ValueError("Flux array shapes don't match")
    # Make sure times match
    if np.any(nuearr[:, 0, 0]-nuebararr[:, 0, 0]>1e-5) or np.any(nuearr[:, 0, 0]-nuebararr[:, 0, 0]>1e-5):
        raise ValueError("Flux arrays do not have mathcing times")

def serialize(
    flux: ParameterizedFlux,
    filename: str,
    groupname:str ="parameterized_flux",
    **kwargs
) ->  None:

    # Make the file / make sure we have write permissions
    from os import path
    if not path.exists(filename):
        with h5.File(filename, "w") as _:
            pass

    # Ensure group name is not already taken
    with h5.File(filename, "r") as h5f:
        idx = 0
        groupname_base = groupname
        groupname = f"{groupname_base}_{idx}"
        while groupname in h5f.keys():
            idx += 1
            groupname = f"{groupname_base}_{idx}"
        # TODO raise warning

    # Make the informationo
    l = len(flux.params)

    ts = np.array([p[0] for p in flux.params])
    ps = np.zeros((3, 4, len(ts)))

    for idx in range(3):
        ps[idx, 0, :] = [p[idx+1].L for p in flux.params]
        ps[idx, 1, :] = [p[idx+1].e_exp for p in flux.params]
        ps[idx, 2, :] = [p[idx+1].rms for p in flux.params]
        ps[idx, 3, :] = [p[idx+1].alpha for p in flux.params]

    # Write the information
    with h5.File(filename, "r+") as h5f:
        h5f.create_group(groupname)
        h5f[groupname]["params"] = ps
        h5f[groupname]["times"] = ts
        h5f[groupname]["distance"] = [flux.dist]
        for k, v in kwargs.items():
            h5f[groupname].attrs[k] = v

def deserialize(g: h5.Group):
    times = g["times"]
    params = []
    for idx, t in enumerate(times):
        pnue = Params(
            g["params"][0, 0, idx],
            g["params"][0, 1, idx],
            g["params"][0, 2, idx],
            g["params"][0, 3, idx]
        )
        pnuebar = Params(
            g["params"][1, 0, idx],
            g["params"][1, 1, idx],
            g["params"][1, 2, idx],
            g["params"][1, 3, idx]
        )
        pnux = Params(
            g["params"][2, 0, idx],
            g["params"][2, 1, idx],
            g["params"][2, 2, idx],
            g["params"][2, 3, idx]
        )
        params.append((t, pnue, pnuebar, pnux))
    return ParameterizedFlux(params, g["distance"][0])

def compute_moment(energies: np.ndarray, flux: np.ndarray, k: int, rtol=1e-7):
    interp = interp1d(energies, np.power(energies, k) * flux)
    yscale = interp(energies.min())
    f = lambda energy: interp(energy) / yscale
    moment, err = quad(
        f,
        energies.min(),
        energies.max(),
        points=energies[::2],
        limit=len(energies)+1
    )
    try:
        if err / moment > rtol:
            raise ValueError(err / moment)
        return moment * yscale
    except Exception as e:
        print(flux)
        raise e

def parameterize(
    nuearr,
    nuebararr,
    nuxarr,
    track=True,
    thin=1
) -> List[Params]:
    slc = slice(None, None, thin)
    nuearr = nuearr[slc]
    nuebararr = nuebararr[slc]
    nuxarr = nuxarr[slc]
    params = []
    itr = range(nuearr.shape[0])
    if track:
        itr = tqdm(itr)
    for idx in itr:
        t = nuearr[idx, 0, 0] * units["second"]
        params_nue = old_compute_params(nuearr[idx])
        params_nuebar = old_compute_params(nuebararr[idx])
        params_nux = old_compute_params(nuxarr[idx])
        params.append((t, params_nue, params_nuebar, params_nux))
    return params

def spectrum(L, e_exp, alpha):
    a = L / e_exp**2
    a *= (alpha + 1)**(alpha+1) / gamma(alpha + 1)
    phi_f = lambda e: a * np.power(e / e_exp, alpha) * np.exp(-(alpha+1) * e / e_exp)
    return phi_f
