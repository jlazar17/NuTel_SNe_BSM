import os
import numpy as np
import h5py as h5

from typing import Dict, Any, List, Optional

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
    sig_results: List,
    bg_results: List,
    basegroupname: str="results",
    metadata:Optional[Dict]=None
):
    if not os.path.exists(fname):
        with h5.File(fname, "w") as _:
            pass

    fields = "bg_sm_norm sb_sm_norm sb_bsm_norm ts".split()
    with h5.File(fname, "r+") as h5f:
        groupname = make_groupname(h5f, basegroupname)
        h5f[groupname].create_group("signal")
        h5f[groupname].create_group("background")
        for field in fields:
            data = [getattr(result, field) for result in bg_results]
            h5f[f"{groupname}/background"].create_dataset(field, data=data)
            data = [getattr(result, field) for result in sig_results]
            h5f[f"{groupname}/signal"].create_dataset(field, data=data)
        if metadata is None:
            return
        add_metadata(h5f[groupname], metadata)
