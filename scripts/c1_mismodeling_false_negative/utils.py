import os
import numpy as np
import h5py as h5
import snewpy

from dataclasses import fields
from typing import Dict, Any, List, Optional
from dataclasses import fields
from asteria.simulation import Simulation

from sne_bsm import units

def get_snewpy_hits():
    from astropy import units as u
    model = {}
    model["name"] = "Nakazato_2013"
    params = getattr(snewpy.models.ccsn, model["name"]).get_param_combinations()[0]
    model["param"] = params
    sim = Simulation(
        model=model,
        distance=10 * u.kpc, 
        Emin=0*u.MeV,
        Emax=100*u.MeV,
        dE=1*u.MeV,
        tmin=-1*u.s,
        tmax=100*u.s,
        dt=0.001*u.s,
        mixing_scheme='AdiabaticMSW',
        hierarchy='normal'
    )
    sim.run()
    ts, hits = sim.detector_signal(dt=0.01 * u.second)
    return ts.value * units["second"], hits

def make_groupname(h5f: h5.File, basegroupname: str) -> None:
    idx = 0
    groupname = basegroupname
    while True:
        if idx > 1000:
            raise RuntimeError("Too many attempts")
        try:
            h5f.create_group(groupname)
            return groupname
        except ValueError:
            idx += 1
            groupname = f"{basegroupname}_{idx}"

def add_metadata(group: h5.Group, metadata: Dict[str, Any]) -> None:
    for k, v in metadata.items():
        group.attrs[k] = v

def save_trials_results(
    fname: str,
    real_results: List,
    fake_results: List,
    basegroupname: str="results",
    metadata:Optional[Dict]=None
):
    if not os.path.exists(fname):
        with h5.File(fname, "w") as _:
            pass

    res = real_results[0]
    fieldnames = [f.name for f in fields(res)]
    with h5.File(fname, "r+") as h5f:
        groupname = make_groupname(h5f, basegroupname)
        h5f[groupname].create_group("real")
        h5f[groupname].create_group("fake")
        for field in fieldnames:
            data = [getattr(result, field) for result in real_results]
            h5f[f"{groupname}/real"].create_dataset(field, data=data)
            data = [getattr(result, field) for result in fake_results]
            h5f[f"{groupname}/fake"].create_dataset(field, data=data)
        if metadata is None:
            return
        add_metadata(h5f[groupname], metadata)
