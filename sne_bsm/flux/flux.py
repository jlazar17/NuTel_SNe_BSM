import numpy as np

from abc import ABC

from ..units import units

class Flux(ABC):
    """Base class for fluxes"""

    def __init__(self):
        pass

    def get_flux(self, t: float, enu: float, nutype: int) -> float:
        """
        Returns the flux at a certain energy at a certain point
        in time
        """
        pass

    def get_hits(
        self,
        tmin=units["second"],
        tmax=20*units["second"],
        dt=0.01*units["second"]
    ) -> np.ndarray:
        pass
