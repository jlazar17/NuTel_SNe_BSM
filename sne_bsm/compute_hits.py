import numpy as np

from tqdm.notebook import tqdm

import os
if "ASTERIA" not in os.environ.keys():
    raise RuntimeError(
        "'ASTERIA' not in os.environ.keys"
    )
os.environ["ASTERIA"] = "/Users/jlazar/research/ASTERIA"

import astropy.units as u

from snewpy.models import ccsn
from asteria.simulation import Simulation

from ic_sn_hnl import units, ParameterizedFlux
