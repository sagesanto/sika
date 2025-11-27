from itertools import product
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import time
import logging

from sika.provider import ContinuousProvider, IntermediateTask, Provider
from .spectrum import Spectrum

__all__ = ["SpectralGridInterpolator"]

class SpectralGridInterpolator(ContinuousProvider[Spectrum], IntermediateTask[Provider[Spectrum]]):
    """ Interpolates between spectral models on a grid of parameters. """
    
    def __init__(self, *args, **kwargs):
        self._keys = None  # establish key order for parameters for later interpolation
        self._grid_interp = None
        self.wlen = None
        self._param_product = None
        super().__init__(*args, **kwargs)
    
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
        vals = [self.prev.provided_parameters[k] for k in self._keys]
        self._param_product = list(product(*vals))
        fluxes = []
        d = {}
        for params in self._param_product:
            spectrum = self.prev(dict(zip(self._keys, params)))
            if self.wlen is None:
                self.wlen = spectrum.wlen
            fluxes.append(spectrum.flux)
            d[spectrum.flux.shape[0]] = params
        if len(np.unique([f.shape for f in fluxes])) > 1:
            self.write_out("Oh no! Some parameters get flux arrays with different lengths!",level=logging.ERROR)
            self.write_out(f"Here are some parameter values that produce different-length arrays: {d}",level=logging.ERROR)
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
        return Spectrum(parameters=parameters, wlen=self.wlen, flux=flux, metadata={"interpolated": True})