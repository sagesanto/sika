from itertools import product
from typing import Dict, List, Any
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import time
import logging
from dataclasses import dataclass

from sika.config.config import Config
from sika.product import Product, FileWritableProduct, ArrayProduct1D
from sika.provider import ContinuousProvider, IntermediateTask, Provider
from sika.task import Task
from .spectra import Spectrum
from sika.utils import parse_path
from .utils import integrate_flux
import pandas as pd

@dataclass(kw_only=True)
class Flux(ArrayProduct1D):
    flux: float
    flux_err: float

    @classmethod
    def value_keywords(cls) -> List[str]:
        return ["flux","flux_err"]
    

class FluxIntegrator(Provider[Flux], IntermediateTask[Provider[Spectrum]]):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    
    @property
    def provided_parameters(self):
        p = self.prev.provided_parameters.copy()
        p["min_wlen"] = []
        p["max_wlen"] = []
        return p
    
    def _call(self, parameters: Dict[str, Any]) -> Flux:
        p_copy = parameters.copy()
        min_wlen = p_copy.pop("min_wlen")
        max_wlen = p_copy.pop("max_wlen")
        spec = self.prev(p_copy)
        f = integrate_flux(spec.wlen,spec.flux,min_wlen, max_wlen)
        err = integrate_flux(spec.wlen, spec.errors,max_wlen, max_wlen)
        
        return Flux(parameters=parameters, flux=f, flux_err=err, metadata={"spectrum_metadata":spec.metadata} ) 
    
    
class FluxGridInterpolator(ContinuousProvider[Flux], IntermediateTask[Provider[Flux]]):
    """ Interpolates between spectral models on a grid of parameters. """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._keys = None  # establish key order for parameters for later interpolation
        self._grid_interp = None
        self._param_product = None

    def args_to_dict(self):
        return {}
    
    # TODO: make sure this preserves param order
    def _setup(self):
        """ loop over the parameter range and load each spectra. then, make a RegularGridInterpolator"""
        # only interpolating the fluxes - wavelengths should be same
        
        if self._grid_interp is not None:
            self.write_out("SpectralGridInterpolator has already been previously set up. Not reloading grid (may cause problems if this occurs because a second, different config was provided!)",level=logging.WARNING)
            return 
        
        self._keys = list(self.prev.provided_parameters.keys())
        
        self._keys = [k for k in self._keys if k not in ["min_wlen","max_wlen"]]
        
        vals = [self.prev.provided_parameters[k] for k in self._keys]
        self._param_product = list(product(*vals))
        fluxes = []
        for params in self._param_product:
            flux_obj = self.prev(dict(zip(self._keys, params)))
            fluxes.append(flux_obj.flux)
            
        fluxes = np.array(fluxes)
        fluxes = fluxes.reshape(*[len(p) for p in vals], -1)  # ensure shape is correct
        self.write_out("Spectral grid flux shape:",fluxes.shape,level=logging.DEBUG)
        self._grid_interp = RegularGridInterpolator((np.array(v) for v in vals), fluxes)

    @property
    def provided_parameters(self):
        return {k: (min(v), max(v)) for k, v in self.prev.provided_parameters.items()}
    
    def _call(self, parameters) -> Spectrum:
        # get the parameter values in the order of the grid
        param_values = [parameters[k] for k in self._keys]

        # interpolate the flux at the given parameters
        try:
            flux = self._grid_interp(param_values)[0,:]
        except ValueError as e:
            raise ValueError(f"Failed to interpolate the spectral grid with parameters {dict(zip(self._keys,param_values))}") from e
        # return a new Spectrum with the interpolated flux and the same wavelengths
        return Flux(parameters=parameters, flux=flux, flux_err=0, metadata={"interpolated": True})