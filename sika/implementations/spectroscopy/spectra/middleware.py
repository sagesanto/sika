from os.path import join, exists
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from sika.provider import Provider, ProviderMiddleware
from sika.implementations.general.metadata import MetadataMerger
from .spectrum import Spectrum
from ..flux import Flux
from sika.config import Config

__all__ = ["PercentileScaler", "PassbandRestrictor", "SpectrumVisualization", "KBandCoupler"]

class PercentileScaler(ProviderMiddleware[Spectrum]):
    """ Scale the flux of the previous spectral model to a specified percentile of its flux. """
    
    def __init__(self, percentile: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.percentile = percentile

    def product_middleware(self, model: Spectrum) -> Spectrum:
        # lol this used to actually perform the normalization but i just made it a method instead
        model.normalize()
        # model.metadata["scale_factor"] = scale_factor
        return model

class PassbandRestrictor(ProviderMiddleware[Spectrum]):
    """ Restrict the wavelength range of the previous spectral model to a specified range. """
    
    def __init__(self, min_wavelength: float, max_wavelength: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength

    def product_middleware(self, model: Spectrum) -> Spectrum:
        model.clip_to_wlen_bounds(self.min_wavelength, self.max_wavelength)
        model.metadata["wavelength_bounds"] = (self.min_wavelength, self.max_wavelength)
        return model

class SpectrumVisualization(ProviderMiddleware[Spectrum]):
    """ Middleware to visualize the output of a spectral model. Does not modify parameters or model output. """
    def product_middleware(self, model: Spectrum) -> Spectrum:
        fig, ax = plt.subplots(figsize=(18, 6))
        
        wavelengths = model.wlen
        flux = model.flux

        ax.plot(wavelengths * 1e4, flux)

        plt.xscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('Wavelength [microns]')
        ax.set_ylabel(r'Planet flux, $F_{\lambda}$ [erg cm$^{-2}$ s$^{-1}$ cm$^{-1}$]')
        
        plt.show()
        
        return model
    
class KBandCoupler(MetadataMerger[Spectrum]):
    def __init__(self, prev: Provider[Spectrum], secondary: Provider[Flux], config=None, logger=None):
        super().__init__(prev, secondary, config, logger)
        
    def merge(self, target: Spectrum, modifier:Flux):
        target.metadata["k_band_flux"] = modifier.flux