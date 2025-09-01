from dataclasses import dataclass
import numpy as np
import pandas as pd


from abc import ABC, abstractmethod
from typing import List
from sika.config import Config
from sika.product import DFProduct
from typing import List

__all__ = ["Spectrum"]

@dataclass(kw_only=True)
class Spectrum(DFProduct):
    """
    A simple spectrum - ``flux`` vs ``wlen``, with optional ``errors``
    """
    
    wlen: np.ndarray  # microns
    flux: np.ndarray
    errors: np.ndarray | None = None

    @classmethod
    def cols(cls):
        """:meta private:"""
        return ['wlen', 'flux', 'errors']

    @classmethod
    def nullable_cols(cls) -> List[str]:
        """:meta private:"""
        return ['errors']
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = np.zeros_like(self.flux)

    def clip_to_wlen_bounds(self, min_wlen, max_wlen):
        mask = (self.wlen >= min_wlen) & (self.wlen <= max_wlen)
        self.wlen = self.wlen[mask]
        self.flux = self.flux[mask]
        return self
    
    def plot(self, ax=None, **kwargs):
        """
        Plot the spectrum on the given axes.
        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        merged_kwargs = {
            "alpha":0.5
        }
        merged_kwargs.update(kwargs)

        ax.plot(self.wlen, self.flux, **merged_kwargs)
        if self.errors is not None:
            ax.fill_between(self.wlen, self.flux - self.errors, self.flux + self.errors, alpha=0.2)

        ax.set_ylabel(r"Flux [erg/s/cm$^2$/cm]", fontsize=12)
        ax.set_xlabel("Wavelength [microns]", fontsize=12)

        ax.minorticks_on()
        ax.tick_params(
            axis="both",
            which="major",
            color="k",
            length=18,
            width=2,
            direction="in",
            labelsize=16,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            color="k",
            length=12,
            width=1,
            direction="in",
            labelsize=16,
        )


