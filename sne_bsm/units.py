from collections import namedtuple

_eV = 1.0
_c = 1.0
_hbar = 1.0
_m = 1.0

Units = namedtuple(
    "Units",
    "eV c hbar MeV erg GeV second m cm kpc"
)

class StringUnits(Units):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        return super().__getitem__(key)

units = StringUnits(
    eV = _eV,
    c = _c,
    hbar = _hbar,
    m = _m,
    MeV = 1e6 * _eV,
    erg = 624_151.0 * 1e6 * _eV,
    GeV = 1e9 * _eV,
    second = _hbar / (6.582119569e-16 * _eV),
    cm = 0.01 * _m,
    kpc = 3.086e19 * _m
)
