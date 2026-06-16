from dataclasses import dataclass
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Sequence
from matplotlib.axes import Axes
import matplotlib.pyplot as plt 
from sika.config import Config
from sika.product import DFProduct

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

    def clip_to_wlen_bounds(self, min_wlen, max_wlen, inplace=True):
        mask = (self.wlen_flat >= min_wlen) & (self.wlen_flat <= max_wlen)
        
        clipped_wlen = self.wlen_flat[mask]
        clipped_flux = self.flux_flat[mask]
        if self.errors is not None:
            clipped_errors = self.errors_flat[mask]
        else:
            clipped_errors = None
        if inplace:
            self.wlen = clipped_wlen
            self.flux = clipped_flux
            self.errors = clipped_errors
            return self
        else:
            md = self.metadata.copy()
            md['clipped_wlen'] = (min_wlen,max_wlen)
            return Spectrum(self.parameters,md,wlen=clipped_wlen,flux=clipped_flux,errors=clipped_errors)
    
    def normalize(self,percentile=90):
        """ Divide this spectrum's flux (and error) by its `percentile` percentile inplace"""
        normfactor = np.nanpercentile(self.flux,percentile)
        if normfactor == 0 or np.isnan(normfactor) or np.isinf(normfactor):
            print("flux:",self.flux)
            print(self)
            raise ValueError(f"Scale factor for percentile {percentile} is {normfactor}, cannot scale flux.")
        self.flux /= normfactor
        if self.errors is not None:
            self.errors /= normfactor
    
    @property
    def wlen_flat(self) -> np.ndarray:
        return self.wlen
    
    @property
    def flux_flat(self) -> np.ndarray:
        return self.flux
    
    @property
    def errors_flat(self) -> np.ndarray:
        return self.errors

    def plot(self, ax:Optional[Axes]=None, shade_errors=True, **kwargs) -> Axes:
        """
        Plot the spectrum on the given Axes if provided, or new Axes if not. returns the Axes.
        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        merged_kwargs = {
            "alpha":0.5
        }
        merged_kwargs.update(kwargs)

        ax.plot(self.wlen, self.flux, **merged_kwargs)
        if self.errors is not None and shade_errors:
            ax.fill_between(self.wlen, self.flux - self.errors, self.flux + self.errors, alpha=0.2,color='gray')

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
        return ax


@dataclass(kw_only=True)
class EchelleOrder(Spectrum):
    def __init__(
        self,
        order:int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.order = order
        self.metadata['order'] = self.order
        

# there's some reason that this doesn't inherit from Product but i dont remember why
class EchelleSpectrum:
    def __init__(self, order_indices: Optional[List[int]]=None, spectra: Optional[List[EchelleOrder]]=None, metadata:Optional[dict] = None, parameters:Optional[dict] = None) -> None:
        self.order_indices: List[int] = order_indices if order_indices is not None else []
        self.spectra: List[EchelleOrder] = spectra if spectra is not None else []
        self.metadata = metadata or {}
        self.parameters = parameters or {}
    
    def add_order(self, order_idx: int, spectrum: EchelleOrder):
        # print(f"[ORDER {order_idx}] Adding spec: {spectrum}")
        self.order_indices.append(order_idx)
        self.spectra.append(spectrum)
    
    @property
    def n_orders(self):
        return len(self.order_indices)
    
    @property
    def orders(self):
        return dict(zip(self.order_indices,self.spectra))
    
    def plot(self, axes:Optional[Sequence[Axes] | np.ndarray]=None, **kwargs):
        """
        Plot the spectrum by plotting each order on its own axis. If an array of axes is provided, will plot onto those. Returns an array of axes.
        """
        
        if axes is None:
            fig, axes = plt.subplots(nrows=self.n_orders, figsize=(12, 4*self.n_orders))
        else:
            assert len(axes) >= self.n_orders, f"If axes are provided, must provide at least as many axes as this spectrum has orders (n_orders = {self.n_orders})"
        merged_kwargs = {
            "alpha":0.5
        }
        
        merged_kwargs.update(kwargs)

        for i, order in self.orders.items():
            order.plot(axes[i],**merged_kwargs)
            axes[i].set_title(f'Order {order.order}')
        
        return axes