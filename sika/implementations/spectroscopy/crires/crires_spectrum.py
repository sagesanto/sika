import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.axes import Axes
from typing import List, Tuple, Optional

from sika.implementations.spectroscopy.spectra.spectrum import Spectrum
from sika.implementations.spectroscopy.utils import clean_and_continuum_subtract



__all__ = ["CRIRESSpectrum"]

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
        wlen_by_order = []
        flux_by_order = []
        error_by_order = []
        norm_constants = []
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
            flux_order, wlen_order, error_order, norm_constant = clean_and_continuum_subtract(
                flux_order, wlen_order, error_order, filter_type=filter_type, filter_size=filter_size, bp_sigma=bp_sigma
            )

            wlen_by_order.append(wlen_order)
            flux_by_order.append(flux_order)
            error_by_order.append(error_order)
            norm_constants.append(norm_constant)
        
        self.metadata["continuum_subtracted"] = True
        
        self.wlen = wlen_by_order
        self.flux = flux_by_order
        self.errors = error_by_order
        self.norm_constants = norm_constants

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
    
    @property
    def wlen_flat(self) -> np.ndarray:
        return np.concatenate(self.wlen)
    
    @property
    def flux_flat(self) -> np.ndarray:
        return np.concatenate(self.flux)
    
    @property
    def errors_flat(self) -> np.ndarray:
        return np.concatenate(self.errors)
    
    def plot(self, ax:Optional[List[Axes]]=None, **kwargs):
        """
        Plot the spectrum on the given axes.
        """
        norders = len(self.wlen)
        if ax is None:
            fig, ax = plt.subplots(nrows=norders, figsize=(12, 4*norders))
        
        merged_kwargs = {
            "alpha":0.5
        }
        
        merged_kwargs.update(kwargs)

        for i in range(norders):
            ax[i].plot(self.wlen[i], self.flux[i], **merged_kwargs)
            if self.errors is not None:
                ax[i].fill_between(self.wlen[i], self.flux[i] - self.errors[i], self.flux[i] + self.errors[i], alpha=0.2)
            ax[i].set_ylabel(r"Flux [arbitrary]", fontsize=12)
            ax[i].minorticks_on()
            ax[i].tick_params(
                axis="both",
                which="major",
                color="k",
                length=18,
                width=2,
                direction="in",
                labelsize=16,
            )
            ax[i].tick_params(
                axis="both",
                which="minor",
                color="k",
                length=12,
                width=1,
                direction="in",
                labelsize=16,
            )
        ax[-1].set_xlabel("Wavelength [microns]", fontsize=12)
        
        return ax