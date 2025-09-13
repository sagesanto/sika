from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from kpicdrp.data import (
    BadPixelMap,
    Background,
    TraceParams,
    DetectorFrame,
    Wavecal,
    Spectrum as KSpectrum,
)

from sika.implementations.spectroscopy import Spectrum, utils

__all__ = ["KPICSpectrum"]

def arr_or_none(data: Optional[list]) -> Optional[np.ndarray]:
    if data is None:
        return None
    return np.array(data)

@dataclass(kw_only=True)
class KPICSpectrum(Spectrum):
    def __init__(
        self,
        *args,
        trace_sigmas: np.ndarray,
        filter_type="median",
        filter_size=100,
        bp_sigma=3,
        masked_ranges: List[Tuple[int, int]] = None,
        orders: Optional[List[int]] = None,
        response_wlen: Optional[np.ndarray] = None,
        response_flux: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.trace_sigmas = trace_sigmas
        wlen_by_order = []
        flux_by_order = []
        error_by_order = []
        del_masks = []

        self.masked_ranges = masked_ranges or []

        if orders is None:
            orders = np.arange(len(self.flux))
        self.metadata["orders"] = orders
        
        self.response_wlen = None
        self.response_flux = None
        
        if response_wlen is not None:
            self.response_wlen = np.array(response_wlen,copy=True)[orders]        
        if response_flux is not None:
            self.response_flux = np.array(response_flux,copy=True)[orders]
            
        # arrays of flux and wlen before deletion
        self.orig_flux = np.array(self.flux,copy=True)[orders]
        self.orig_wlen = np.array(self.wlen,copy=True)[orders]

        for o in orders:
            del_mask = np.zeros_like(self.wlen[o])
            for start_wlen, end_wlen in self.masked_ranges:
                del_mask[(self.wlen >= start_wlen) & (self.wlen <= end_wlen)] = 1
            del_mask = del_mask.astype(bool)
            wlen_order = self.wlen[o]
            flux_order = self.flux[o]
            error_order = (
                self.errors[o] if self.errors is not None else np.zeros_like(flux_order)
            )
            # dont want to delete from trace_sigma_order
            # trace_sigma_order = self.trace_sigmas[o]

            wlen_order = np.delete(wlen_order, del_mask)
            flux_order = np.delete(flux_order, del_mask)
            error_order = np.delete(error_order, del_mask)
            # trace_sigma_order = np.delete(trace_sigma_order, del_mask)

            # clean and normalize the spectrum for this order
            flux_order, wlen_order, error_order, norm_constant = (
                utils.clean_and_normalize_spectrum(
                    flux_order,
                    wlen_order,
                    error_order,
                    filter_type=filter_type,
                    filter_size=filter_size,
                    bp_sigma=bp_sigma,
                )
            )
            wlen_by_order.append(wlen_order)
            flux_by_order.append(flux_order)
            error_by_order.append(error_order)
            # trace_sigma_by_order.append(trace_sigma_order)
            del_masks.append(del_mask)
                
        self.del_masks = np.array(del_masks)
        self.flux = flux_by_order
        self.wlen = wlen_by_order
        self.errors = error_by_order
        self.trace_sigmas = np.array(trace_sigmas)[orders]

    @property
    def wlen_flat(self) -> np.ndarray:
        return np.concatenate(self.wlen)
    
    @property
    def flux_flat(self) -> np.ndarray:
        return np.concatenate(self.flux)
    
    @property
    def errors_flat(self) -> np.ndarray:
        return np.concatenate(self.errors)