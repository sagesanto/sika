from dataclasses import dataclass
import numpy as np
import pandas as pd

from sika.implementations.spectroscopy.utils import clean_and_normalize_spectrum

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from sika.config import Config
from sika.product import DFProduct
from typing import List, Tuple

@dataclass(kw_only=True)
class Spectrum(DFProduct):
    wlen: np.ndarray  # microns
    flux: np.ndarray
    errors: np.ndarray | None = None

    @classmethod
    def cols(cls):
        return ['wlen', 'flux', 'errors']

    @classmethod
    def nullable_cols(cls) -> List[str]:
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


@dataclass(kw_only=True)
class CRIRESSpectrum(Spectrum):
    """
    Inherits from Spectrum and is used to handle CRIRES-specific spectral data.
    wlen, flux, and error are input as 1d np arrays and will be reshaped to a list of arrays by spectral order.
    each order is calibrated and normalized separately.
    each CRIRESSpectrum object represents a single night of data
    """

    def __init__(self, *args, order_indices=None, filter_type='median', filter_size=100, bp_sigma=3, masked_ranges:List[Tuple[int,int]]=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.order_indices = order_indices or self.find_order_indices()
        self.norders = len(self.order_indices)
        self.wlen_by_order = []
        self.flux_by_order = []
        self.error_by_order = []
        self.norm_constants = []
        self.masked_ranges = masked_ranges or []
        
        del_mask = np.zeros_like(self.wlen)
        for (start_wlen, end_wlen) in self.masked_ranges:
            del_mask[(self.wlen >= start_wlen) & (self.wlen <= end_wlen)] = 1
        del_mask = del_mask.astype(bool)
        
        for indices in self.order_indices:
            wlen_order = self.wlen[indices]
            flux_order = self.flux[indices]
            error_order = self.errors[indices] if self.errors is not None else np.zeros_like(flux_order)
            mask = del_mask[indices]
            
            wlen_order = np.delete(wlen_order, mask)
            flux_order = np.delete(flux_order, mask)
            error_order = np.delete(error_order, mask)
            # clean and normalize the spectrum for this order
            flux_order, wlen_order, error_order, norm_constant = clean_and_normalize_spectrum(
                flux_order, wlen_order, error_order, filter_type=filter_type, filter_size=filter_size, bp_sigma=bp_sigma
            )
            
            self.wlen_by_order.append(wlen_order)
            self.flux_by_order.append(flux_order)
            self.error_by_order.append(error_order)
            self.norm_constants.append(norm_constant)

    def find_order_indices(self):
        indices = []
        diffs = np.diff(self.wlen)
        diffs /= np.median(diffs)
        ind_edge = np.argwhere(diffs>100).flatten()
        if ind_edge.size > 0:
            ind_edge = np.insert(ind_edge+1, 0, 0)
            ind_edge = np.insert(ind_edge, len(ind_edge), len(self.wlen))
            Nchip = len(ind_edge)-1
            for i in range(Nchip):
                indices.append(np.arange(ind_edge[i], ind_edge[i+1]))
        else:
            indices.append(np.arange(len(self.wlen)))
        
        return indices