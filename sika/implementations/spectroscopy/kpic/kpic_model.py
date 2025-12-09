from typing import TypeVar
import numpy as np

from kpicdrp.xcorr import convolve_and_sample

from sika.modeling.data import Dataset
from sika import ProviderMiddleware, Provider, Product
from sika.modeling import Model, CompositeModel, EmptyParameterSet, Dataset, DataLoader
from sika.implementations.spectroscopy import Spectrum
from .kpic_spectrum import KPICSpectrum, KPICOrder


def model_to_kpic_grid(model:Spectrum, kpic_spec: KPICOrder):    
    convolved_flux = convolve_and_sample(np.array(kpic_spec.orig_wlen),np.array(kpic_spec.trace_sigmas),np.array(model.wlen), np.array(model.flux))
    model.flux = np.array(convolved_flux)
    model.wlen = np.array(kpic_spec.orig_wlen)
    model.metadata["aligned_to_kpic"] = True
    return model

def apply_kpic_response(model:Spectrum, kpic_spec: KPICOrder):
    if kpic_spec.response_flux is None:
        raise ValueError("KPIC spectrum not properly initialized with response. Set the response file in the config for each night's data.")
    model.flux *= np.array(kpic_spec.response_flux)
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
    
    @property
    def provided_parameters(self):
        if self.data_provider:
            return self.data_provider.provided_parameters
        return {}

class KPICModel(CompositeModel[Spectrum]):
    def __init__(self, name:str, spectral_model: Model[Spectrum], kpic_data: Dataset[KPICOrder] | DataLoader[KPICOrder], *args, data_params={}, **kwargs):
        super().__init__(name, EmptyParameterSet(), *args, **kwargs)
        self.spectral_model = spectral_model
        self.data_wrapper = DataWrapper(data_or_loader=kpic_data)
        self.data_params = data_params
        self.data: Dataset[KPICOrder] = None

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
            model = model_set.values(sel).copy()
            
            # in keeping with kpic DRP forward model (https://github.com/kpicteam/kpic_pipeline/blob/main/kpicdrp/xcorr.py#L360),
            # we normalize flux, convolve to data resolution using the LSF,
            # align to the data grid, and then normalize again (because convolution does not conserve flux):
            model.normalize(percentile=90)  # normalize
            model = model_to_kpic_grid(model, kpic_spec)  # convolve and align
            model.normalize(percentile=90)  # normalize
            model = apply_kpic_response(model, kpic_spec)  # apply response
            model.metadata.update(sel)  # so that the dataset knows what fiber/night/etc this spectrum corresponds to
            kpic_models.append(model)
            
        return Dataset(kpic_models, dims=self.data.dims)