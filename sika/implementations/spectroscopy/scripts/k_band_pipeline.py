import sys, os
import matplotlib.pyplot as plt
from os.path import join, abspath, splitext, basename

import json

import numpy as np

from sika.implementations.general.product_iterators import ListIterator
from sika.implementations.spectroscopy import (
    Spectrum,
    CRIRESSpectrum,
    PassbandRestrictor,
    PercentileScaler,
    CommonLogPressureGrid,
    ElfOwl,
    pRT,
    Flux,
    FluxGridInterpolator,
    FluxIntegrator,
    KBandCoupler,
    SpectralGridInterpolator,
)
from sika.implementations.general import FileCache, ParamRestrictor, ParamInjector
from sika.implementations.spectroscopy.utils import plot_spectra, pressure_grid_tiers
from sika.modeling import (
    Parameter,
    DataLoader,
    Dataset,
    Constraint,
)

from sika.modeling.priors import Normal, NullPriorTransform, Uniform
from sika.utils import (
    parse_path,
    write_out as _write_out,
    savefig,
    get_pool,
    file_timestamp,
    current_dt_utc,
)

from sika.config import Config, configure_logger, config_path
import matplotlib


config = Config(config_path)

target_cfg = config[config["target"]]
data_cfg = target_cfg["data"]
wavelen_fudge_factor = config["wavelen_bounds_fudge"]
data_wbounds = data_cfg["wavelen_range"]


WAVELENGTH_BOUNDS = (
    data_wbounds[0] * (1 - wavelen_fudge_factor),
    data_wbounds[1] * (1 + wavelen_fudge_factor),
)

restricted_params = {
    "teff": (700.0, 750.0, 800.0),
    "grav": (178.0,),
    "mh": (0.0,),
    "co": (1.0,),
    "logkzz": (2.0,),
}

spectra = ParamRestrictor(
    allowed_parameters=restricted_params,
    prev=FileCache(
        pRT(CommonLogPressureGrid(ElfOwl())),
        target_cls=Spectrum,
        savedir=abspath(
            join(
                parse_path(config["filestore"]),
                "spectral_models",
                "pRT",
                "elf_owl",
            )
        ),
    ),
)

k_band = FluxGridInterpolator(
    ParamInjector(
        inject={"min_wlen": 2.025, "max_wlen": 2.15},
        prev=FileCache(
            prev=FluxIntegrator(spectra),
            target_cls=Flux,
            savedir=abspath(
                join(
                    parse_path(config["filestore"]),
                    "spectral_models",
                    "pRT",
                    "k_band_fluxes",
                )
            ),
        ),
    )
)

prod = KBandCoupler(
    prev=SpectralGridInterpolator(
        PercentileScaler(
            percentile=90,
            prev=PassbandRestrictor(
                min_wavelength=WAVELENGTH_BOUNDS[0],
                max_wavelength=WAVELENGTH_BOUNDS[1],
                prev=spectra,
            ),
        )
    ),
    secondary=k_band,
)


prod.configure(config, None)

prod.visualize("K-Band Pipeline")
plt.show()
test_params = {k:v[0] for k,v in restricted_params.items()}
s = prod(test_params)
# print(s)
# print(s.metadata)