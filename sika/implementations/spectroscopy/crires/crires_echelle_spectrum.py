import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.axes import Axes
from typing import List, Tuple, Optional, Sequence

from sika.implementations.spectroscopy.spectra.spectrum import Spectrum, EchelleOrder, EchelleSpectrum
from sika.implementations.spectroscopy.utils import clean_and_continuum_subtract, clean, continuum_subtract

__all__ = ["CRIRESEchelleSpectrum","CRIRESOrder"]

@dataclass(kw_only=True)
class CRIRESOrder(EchelleOrder):
    """One order of a CRIRES+ spectrum. This may also represent one detector's portion of one order of a CRIRES+ spectrum (1/3 of an order)"""
    def __init__(self,*args, **kwargs):
        super().__init__(*args,**kwargs)
    
    def plot(self, ax:Optional[Axes]=None, shade_errors=True, **kwargs):
        """
        Plot the spectrum. Will plot onto provided Axes object if given. Returns Axes.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
            
        merged_kwargs = {
            "alpha":0.5
        }
        
        merged_kwargs.update(kwargs)

        ax.plot(self.wlen, self.flux, **merged_kwargs)
        if self.errors is not None and shade_errors:
            ax.fill_between(self.wlen, self.flux - self.errors, self.flux + self.errors, alpha=0.2)
        ax.set_ylabel(r"Flux [arbitrary]", fontsize=12)
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
        ax.set_xlabel("Wavelength [microns]", fontsize=12)
                
        return ax


@dataclass(kw_only=True)
class CRIRESEchelleSpectrum(EchelleSpectrum):
    """
    Handle CRIRES+-specific spectral data.
    wlen, flux, and error are input as 1d np arrays and will be reshaped into :py:class:`~CRIRESOrder` objects.
    each order is calibrated and normalized separately.
    each CRIRESEchelleSpectrum object represents a single night of data
    """

    def __init__(self, wlen:np.ndarray, flux:np.ndarray, errors:np.ndarray,
                    order_indices:Sequence[int]=None, 
                    sigma_clip=True,
                    remove_continuum = True,
                    filter_type:str='median', 
                    filter_size:int=100, 
                    bp_sigma:float=3, 
                    masked_ranges:List[Tuple[float,float]]=None, 
                    metadata: dict | None = None, 
                    **kwargs
                ):
        """A spectrum that consists of multiple echelle orders. Flux, wlen, errors should be provided as one 1d array and will be split, using order indices (if provided) or by order detection if not. 

        :param wlen: 1D (concatenated) wavelength values
        :type wlen: np.ndarray
        :param flux: 1D (concatenated) flux values
        :type flux: np.ndarray
        :param errors: 1D (concatenated) flux errors, in the same units as flux 
        :type errors: np.ndarray
        :param order_indices: autodetected if not specified. a list of lists of indices that specify which flux/wlen/error values belong in which orders.
        :type order_indices: List[Sequence[int]], optional
        :param sigma_clip: whether to clean the continuum with a sigma clip of sigma=bp_sigma, defaults to True
        :type sigma_clip: bool, optional
        :param remove_continuum: whether to remove the continuum using the specified filter, defaults to True
        :type remove_continuum: bool, optional
        :param filter_type: filter type for continuum removal, defaults to 'median'
        :type filter_type: str, optional
        :param filter_size: filter size for continuum removal, defaults to 100
        :type filter_size: int, optional
        :param bp_sigma: sigma threshold to use for outlier flux removal, defaults to 3
        :type bp_sigma: float, optional
        :param masked_ranges: collection of (start, stop) tuples of masked-out wavelength ranges to remove
        :type masked_ranges: List[Tuple[float,float]], optional
        :param metadata: dictionary of metadata to attach to each loaded :py:class:`~CRIRESOrder`
        :type metadata: dict[str,Any]
        """
        super().__init__(order_indices=[], spectra=[], metadata=metadata, **kwargs)
        # this metadata will be shared by the orders
        if metadata is not None:
            metadata = metadata.copy()
        else:
            metadata = {}
        
        order_indices = order_indices or self.find_order_indices(wlen)
        wlen_by_order = []
        flux_by_order = []
        error_by_order = []
        norm_constants = []
        self.masked_ranges = masked_ranges or []

        del_mask = np.zeros_like(wlen)
        for (start_wlen, end_wlen) in self.masked_ranges:
            del_mask[(wlen >= start_wlen) & (wlen <= end_wlen)] = 1
        del_mask = del_mask.astype(bool)
        
        if remove_continuum:
            metadata['continuum_subtracted'] = dict(filter_type=filter_type, filter_size=filter_size)
        else:
            metadata['continuum_subtracted'] = False
        if sigma_clip:
            metadata['bp_sigma'] = bp_sigma
            
        # print(f"{wlen.shape = }")
        # print(f"{flux.shape = }")
        
        for i, indices in enumerate(order_indices):
            o_md = metadata.copy()
            
            wlen_order = wlen[indices]
            flux_order = flux[indices]
            error_order = errors[indices] if errors is not None else np.zeros_like(flux_order)
            
            mask = del_mask[indices]
            if np.any(mask):
                o_md['masked_regions_removed'] = True
            wlen_order = np.delete(wlen_order, mask)
            flux_order = np.delete(flux_order, mask)
            error_order = np.delete(error_order, mask)
            
            # clean and normalize the spectrum for this order
            if sigma_clip:
                flux_order, wlen_order, error_order = clean(flux_order,wlen_order,error_order,bp_sigma)
            if remove_continuum:
                flux_order, norm_constant = continuum_subtract(flux_order, filter_size, filter_type)
            else:
                norm_constant = None
            o_md['norm_constant'] = norm_constant
            
            ninetieth = np.nanpercentile(flux_order,90)
            o_md['pre_norm_90th_percentile'] = ninetieth
            if ninetieth != 0:
                flux_order /= ninetieth
                error_order /= ninetieth
            else:
                print(f'Warning! 90th percentile of order {i} of spectrum with metadata {metadata} is zero!')
                
            order_spec = CRIRESOrder(
                parameters={},
                order=i,
                wlen=wlen_order,
                flux=flux_order,
                errors=error_order,
                metadata=o_md
            )
            
            self.add_order(i,order_spec)

            wlen_by_order.append(order_spec.wlen)
            flux_by_order.append(order_spec.flux)
            error_by_order.append(order_spec.errors)
            norm_constants.append(norm_constant)
        
        self.wlen = wlen_by_order
        self.flux = flux_by_order
        self.errors = error_by_order
        self.norm_constants = norm_constants

    def find_order_indices(self, wlen):
        indices = []
        diffs = np.diff(wlen)
        diffs /= np.median(diffs)
        ind_edge = np.argwhere(diffs>100).flatten()
        # print(f"{ind_edge = }")
        # print(f"{wlen = }")
        # print('ind_edge:',ind_edge)
        if ind_edge.size > 0:
            ind_edge = np.insert(ind_edge+1, 0, 0)
            ind_edge = np.insert(ind_edge, len(ind_edge), len(wlen))
            # print('ind_edge after insert:',ind_edge)
            Nchip = len(ind_edge)-1
            # print(f"{Nchip = }")
            for i in range(Nchip):
                indices.append(np.arange(ind_edge[i], ind_edge[i+1]))
        else:
            indices.append(np.arange(len(wlen)))

        return indices
    
    @property
    def wlen_flat(self) -> np.ndarray:
        return np.concatenate([s.wlen for s in self.spectra])
    
    @property
    def flux_flat(self) -> np.ndarray:
        return np.concatenate([s.flux for s in self.spectra])
    
    @property
    def errors_flat(self) -> np.ndarray:
        return np.concatenate([s.errors for s in self.spectra])