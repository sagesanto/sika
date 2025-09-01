from typing import TypeVar
import numpy as np

from kpicdrp.xcorr import convolve_and_sample

from sika.modeling.data import Dataset
from sika import ProviderMiddleware, Provider, Product
from sika.modeling import Model, CompositeModel, EmptyParameterSet, Dataset, DataLoader
from sika.implementations.spectroscopy import Spectrum
from .kpic_spectrum import KPICSpectrum


def model_to_kpic_grid(model:Spectrum, kpic_spec: KPICSpectrum):
        convolved_fluxes = []
        for i in range(len(kpic_spec.orig_wlen)):
            convolved_fluxes.append(convolve_and_sample(np.array(kpic_spec.orig_wlen[i]),np.array(kpic_spec.trace_sigmas[i]),np.array(model.wlen), np.array(model.flux)))
        model.flux = np.array(convolved_fluxes).flatten()
        model.wlen = np.array(kpic_spec.orig_wlen).flatten()
        
        model.metadata["aligned_to_kpic"] = True
        return model

def apply_kpic_response(model:Spectrum, kpic_spec: KPICSpectrum):
    if kpic_spec.response_flux is None:
        raise ValueError("KPIC spectrum not properly initialized with response. Set the response file in the config for each night's data.")
    model.flux *= kpic_spec.response_flux.flatten()
    return model

D = TypeVar('D', bound=Product, covariant=True)
class DataWrapper(Provider[Dataset[D]]):
    """ Provides a unified interface for loading data from either one already-loaded Dataset or from a DataLoader"""
    def __init__(self, *args, data_or_loader: Dataset[D] | DataLoader[D], **kwargs):
        self.data = None
        self.data_provider = None
        if isinstance(data_or_loader, Dataset):
            self.data = data_or_loader
        else:
            self.data_provider = data_or_loader
        super().__init__(*args, prev=self.data_provider, **kwargs)
    
    def _call(self, parameters) -> Dataset[D]:
        if self.data:
            return self.data
        elif self.data_provider:
            return self.data_provider(parameters)
        raise ValueError("No data available")

class KPICModel(CompositeModel[Spectrum]):
    def __init__(self,spectral_model: Model[Spectrum], kpic_data: Dataset[D] | DataLoader[D], *args, data_params={}, **kwargs):
        super().__init__(EmptyParameterSet(), *args, **kwargs)
        self.spectral_model = spectral_model
        self.data_wrapper = DataWrapper(data_or_loader=kpic_data)
        self.data_params = data_params
        self.data: Dataset[KPICSpectrum] = None

    def _setup(self):
        self.data = self.data_wrapper(self.data_params)
    
    @property
    def models(self):
        return [self.spectral_model]
    
    @property
    def previous(self):
        return [self.spectral_model, self.data_wrapper]
    
    def make_model(self) -> Dataset[Spectrum]:    
        model_set = self.spectral_model.make_model()

        kpic_models = []
        for sel, kpic_spec in self.data:
            model = model_set.values(sel)
            model = model_to_kpic_grid(model, kpic_spec)
            model = apply_kpic_response(model, kpic_spec)
            model.metadata.update(sel)  # so that the dataset knows what fiber/night/etc this spectrum corresponds to
            kpic_models.append(model)
        
        return Dataset(kpic_models, dims=self.data.dims)