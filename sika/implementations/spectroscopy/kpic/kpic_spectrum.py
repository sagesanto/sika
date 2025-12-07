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

from sika.implementations.spectroscopy import Spectrum, utils, EchelleSpectrum, EchelleOrder

__all__ = ["KPICSpectrum"]

def arr_or_none(data: Optional[list]) -> Optional[np.ndarray]:
    if data is None:
        return None
    return np.array(data)

@dataclass(kw_only=True)
class KPICOrder(EchelleOrder):
    def __init__(self,*args,orig_wlen:Optional[List[float]]=None,trace_sigmas:Optional[List[float]]=None,response_flux:Optional[List[float]]=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.orig_wlen = orig_wlen
        self.trace_sigmas = trace_sigmas
        self.response_flux = response_flux

class KPICSpectrum(EchelleSpectrum):
    def __init__(self, wlen, flux, errors, 
                    trace_sigmas: np.ndarray,
                    filter_type="median",
                    filter_size=100,
                    bp_sigma=3,
                    masked_ranges: List[Tuple[int, int]] = None,
                    orders: List[int] | None = None, 
                    response_wlen: Optional[np.ndarray] = None,
                    response_flux: Optional[np.ndarray] = None,
                    metadata: dict | None = None
                ) -> None:
        super().__init__(order_indices=[], spectra=[], metadata=metadata)
        self.wlen = wlen
        self.flux = flux
        self.errors = errors
        metadata = metadata or {}
        
        self.trace_sigmas = trace_sigmas
        wlen_by_order: List[List[float]] = []
        flux_by_order: List[List[float]] = []
        error_by_order: List[List[float]] = []
        del_masks: List[List[int]] = []

        self.masked_ranges = masked_ranges or []

        if orders is None:
            orders = np.arange(len(self.flux))
        # self.metadata["orders"] = orders
        
        self.response_wlen = None
        self.response_flux = None
        
        if response_wlen is not None:
            self.response_wlen = np.array(response_wlen,copy=True)[orders]        
        if response_flux is not None:
            self.response_flux = np.array(response_flux,copy=True)[orders]
            
        # arrays of flux and wlen before deletion
        self.orig_flux = np.array(self.flux,copy=True)[orders]
        self.orig_wlen = np.array(self.wlen,copy=True)[orders]

        for i, o in enumerate(orders):
            wlen_order = np.array(self.wlen[o], copy=True)
            flux_order = np.array(self.flux[o], copy=True)
            del_mask = np.zeros_like(wlen_order)
            for start_wlen, end_wlen in self.masked_ranges:
                del_mask[(wlen_order >= start_wlen) & (wlen_order <= end_wlen)] = 1
            del_mask = del_mask.astype(bool)
            error_order = np.array(
                self.errors[o] if self.errors is not None else np.zeros_like(flux_order), copy=True
            )
            # dont want to delete from trace_sigma_order
            # trace_sigma_order = self.trace_sigmas[o]

            wlen_order = np.delete(wlen_order, del_mask)
            flux_order = np.delete(flux_order, del_mask)
            error_order = np.delete(error_order, del_mask)
            # trace_sigma_order = np.delete(trace_sigma_order, del_mask)

            # clean and normalize the spectrum for this order
            flux_order, wlen_order, error_order, norm_constant = (
                utils.clean_and_continuum_subtract(
                    flux_order,
                    wlen_order,
                    error_order,
                    filter_type=filter_type,
                    filter_size=filter_size,
                    bp_sigma=bp_sigma,
                )
            )
            order_spec = KPICOrder(
                parameters={},
                orig_wlen=np.array(self.orig_wlen[i]),
                trace_sigmas=np.array(trace_sigmas)[o],
                response_flux=np.array(self.response_flux[i]) if self.response_flux is not None else None,
                order=o,
                wlen=wlen_order,
                flux=flux_order,
                errors=error_order,
                metadata=metadata.copy()
            )
            self.add_order(o, order_spec)
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


# @dataclass(kw_only=True)
# class KPICSpectrum(Spectrum):
#     def __init__(
#         self,
#         *args,
#         trace_sigmas: np.ndarray,
#         filter_type="median",
#         filter_size=100,
#         bp_sigma=3,
#         masked_ranges: List[Tuple[int, int]] = None,
#         orders: Optional[List[int]] = None,
#         response_wlen: Optional[np.ndarray] = None,
#         response_flux: Optional[np.ndarray] = None,
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         self.trace_sigmas = trace_sigmas
#         wlen_by_order = []
#         flux_by_order = []
#         error_by_order = []
#         del_masks = []

#         self.masked_ranges = masked_ranges or []

#         if orders is None:
#             orders = np.arange(len(self.flux))
#         self.metadata["orders"] = orders
        
#         self.response_wlen = None
#         self.response_flux = None
        
#         if response_wlen is not None:
#             self.response_wlen = np.array(response_wlen,copy=True)[orders]        
#         if response_flux is not None:
#             self.response_flux = np.array(response_flux,copy=True)[orders]
            
#         # arrays of flux and wlen before deletion
#         self.orig_flux = np.array(self.flux,copy=True)[orders]
#         self.orig_wlen = np.array(self.wlen,copy=True)[orders]

#         for o in orders:
#             del_mask = np.zeros_like(self.wlen[o])
#             for start_wlen, end_wlen in self.masked_ranges:
#                 del_mask[(self.wlen >= start_wlen) & (self.wlen <= end_wlen)] = 1
#             del_mask = del_mask.astype(bool)
#             wlen_order = self.wlen[o]
#             flux_order = self.flux[o]
#             error_order = (
#                 self.errors[o] if self.errors is not None else np.zeros_like(flux_order)
#             )
#             # dont want to delete from trace_sigma_order
#             # trace_sigma_order = self.trace_sigmas[o]

#             wlen_order = np.delete(wlen_order, del_mask)
#             flux_order = np.delete(flux_order, del_mask)
#             error_order = np.delete(error_order, del_mask)
#             # trace_sigma_order = np.delete(trace_sigma_order, del_mask)

#             # clean and normalize the spectrum for this order
#             flux_order, wlen_order, error_order, norm_constant = (
#                 utils.clean_and_normalize_spectrum(
#                     flux_order,
#                     wlen_order,
#                     error_order,
#                     filter_type=filter_type,
#                     filter_size=filter_size,
#                     bp_sigma=bp_sigma,
#                 )
#             )
#             wlen_by_order.append(wlen_order)
#             flux_by_order.append(flux_order)
#             error_by_order.append(error_order)
#             # trace_sigma_by_order.append(trace_sigma_order)
#             del_masks.append(del_mask)
#         self.norders = len(flux_by_order)
#         self.del_masks = np.array(del_masks)
#         self.flux = flux_by_order
#         self.wlen = wlen_by_order
#         self.errors = error_by_order
#         self.trace_sigmas = np.array(trace_sigmas)[orders]

#     @property
#     def wlen_flat(self) -> np.ndarray:
#         return np.concatenate(self.wlen)
    
#     @property
#     def flux_flat(self) -> np.ndarray:
#         return np.concatenate(self.flux)
    
#     @property
#     def errors_flat(self) -> np.ndarray:
#         return np.concatenate(self.errors)