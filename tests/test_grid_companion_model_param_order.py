import pytest
import numpy as np
from sika.modeling.priors import Uniform
from sika.modeling.params import Parameter
from sika.implementations.spectroscopy.grid_companion_model import ElfOwlCompanionParameterSet
from sika.utils import groupby

def test_eo_paramset_rv_order_consistency():
    # RV varies with night, others are constant
    nights = ["n1", "n2", "n3"]
    
    RVs = [1, 2, 3]
    vsini = 5
    teff = 2500
    grav = 4
    mh = 0
    co = 0
    logkzz = 5
    
    rv_param = Parameter("rv", Uniform(-100, 100), varies_with=["night"], coords={"night": nights})
    vsini_param = Parameter("vsini", Uniform(0, 20))
    teff_param = Parameter("teff", Uniform(2000, 3000))
    grav_param = Parameter("grav", Uniform(3, 5))
    mh_param = Parameter("mh", Uniform(-1, 1))
    co_param = Parameter("co", Uniform(-1, 1))
    logkzz_param = Parameter("logkzz", Uniform(0, 10))

    pset = ElfOwlCompanionParameterSet(
        name="test",
        rv=rv_param,
        vsini=vsini_param,
        teff=teff_param,
        grav=grav_param,
        mh=mh_param,
        co=co_param,
        logkzz=logkzz_param,
    )
    pset.set_values_flat(RVs + [vsini, teff, grav, mh, co, logkzz])
    all_param_names = [p.name for p in pset.params]
    non_night_params = [p for p in all_param_names if p != "rv"]
    # groupby non-night params (which are all constant)
    
    for spectra_params, remaining_params in pset.groupby(
            ["teff","grav","mh","co","logkzz"]):
    
        for vsini_dict, coords, rvs in groupby(
                ["vsini"], remaining_params, flatten=True
            ):
            assert vsini == vsini_dict["vsini"]
            for coord, rv_dict in zip(coords, rvs):
                assert rv_dict["rv"] == RVs[nights.index(coord['night'])]
