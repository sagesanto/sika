from typing import TypeVar, Optional
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

from sika.modeling.data import Dataset
from sika import ProviderMiddleware, Provider, Product
from sika.modeling import Model, CompositeModel, EmptyParameterSet, Dataset, DataLoader, DataWrapper
from sika.implementations.spectroscopy import Spectrum
from .crires_echelle_spectrum import CRIRESOrder

# adapted from jerry xuan
def scale_model_to_order(order_wlen: np.ndarray, model_wlen: np.ndarray, model_flux:np.ndarray, filter_type, filter_size, flux_errors=None) -> np.ndarray:
    # take a model representing the entire spectrum and crop/scale it to a specific order in the data
    # performs continuum subtraction with a filter of type 'filter_type' and of size 'filter_size'
    # returns the scaled flux array for that order
    f = np.interp(order_wlen, model_wlen, model_flux)
    if flux_errors is not None and not np.all(flux_errors==0):
        e = np.interp(order_wlen, model_wlen, flux_errors)
    else:
        e = np.zeros_like(f)
    nans_in_interp = len(np.where(np.isnan(f))[0])
    if nans_in_interp == len(f):
        print("WARNING: interpolating the model flux to the order wavelength produced all NaNs. This likely means that the model wavelength range does not overlap with the order wavelength range at all, or that the model flux contains many NaNs. Continuing, but this will likely cause problems later.")
        print("order wlen bounds:", order_wlen.min(), order_wlen.max())
        print("model wlen bounds:", model_wlen.min(), model_wlen.max())
        print("order wlen shape:", order_wlen.shape)
        print("model wlen shape:", model_wlen.shape)
        print("model flux shape:", model_flux.shape)
        print("model:", np.c_[model_wlen, model_flux])
        between = (model_wlen >= order_wlen.min()) & (model_wlen <= order_wlen.max())
        print("model between min/max of order:", np.c_[model_wlen[between], model_flux[between]])
    # print("interpolated flux shape:", f.shape)
    bad = np.where(np.isnan(f))  # get bad indices
    f[bad] = np.nanmedian(f)
    if filter_type == 'median':
        continuum = ndi.median_filter(f, filter_size)
    elif filter_type == 'gaussian':
        continuum = ndi.gaussian_filter(f, filter_size)
    else:
        raise ValueError(f"Filter type {filter_type} not recognized. Use 'median' or 'gaussian'.")
    f = f - continuum
    f[bad] = np.nan
    e[bad] = np.nan
    # print("----- done scaling -------")
    return f, e

class CRIRESModel(CompositeModel[Spectrum]):
    def __init__(self, name:str, spectral_model: Model[Spectrum], crires_data: Dataset[CRIRESOrder] | DataLoader[CRIRESOrder], *args, data_params:Optional[dict]=None, **kwargs):
        super().__init__(name, EmptyParameterSet(), *args, **kwargs)
        self.spectral_model = spectral_model
        self.data_wrapper = DataWrapper(data_or_loader=crires_data)
        self.data_params = data_params or {}
        self.data: Dataset[CRIRESOrder] = None

    def _setup(self):
        self.data = self.data_wrapper(self.data_params)
        
        target_name = self.config["target"]
        target_cfg = self.config[target_name]
        data_cfg = target_cfg["data"]
        self.merged_cfg = dict(data_cfg).copy()
        self.merged_cfg.update(self.data_params)
        
        self.filter_size = self.merged_cfg["filter_size"]
        self.filter_type = self.merged_cfg["filter_type"]
        self.bp_sigma = self.merged_cfg["bp_sigma"]
    
    @property
    def models(self):
        return [self.spectral_model]
    
    @property
    def previous(self):
        return [self.spectral_model, self.data_wrapper]
    
    def make_model(self) -> Dataset[CRIRESOrder]:    
        model_set = self.spectral_model.make_model()

        crires_models = []
        for sel, data_spec in self.data:
            model = model_set.values(sel).copy()
            
            # print(f'{data_spec.wlen.shape = }')
            # print(f'{data_spec.flux.shape = }')
            # print(f'{model.wlen.shape = }')
            # print(f'{model.flux.shape = }')
            
            # wlen_min, wlen_max = min(data_spec.wlen), max(data_spec.wlen)
            # fig, axes = plt.subplots(nrows=3, figsize=(20,7),sharex=True)
            # fig.tight_layout()
            # fig.suptitle(f"{self.name} - {sel}",y=1.07)
            # m_w, m_f = np.copy(model.wlen), np.copy(model.flux)  # get a copy of original model vals for plotting
            
            # interp_errors = model.errors is not None and not np.all(model.errors==0)
            model.flux, model.errors = scale_model_to_order(data_spec.wlen, model.wlen_flat,model.flux_flat, self.filter_type, self.filter_size, flux_errors=model.errors)
            model.wlen = data_spec.wlen
            # model.errors = np.zeros_like(model.wlen)
                
            model.normalize(percentile=90)
      
      
            # in keeping with kpic DRP forward model (https://github.com/kpicteam/kpic_pipeline/blob/main/kpicdrp/xcorr.py#L360),
            # we normalize flux, convolve to data resolution using the LSF,
            # align to the data grid, and then normalize again (because convolution does not conserve flux):
            
            # ax = axes[1]
            # ax.set_title("Aligned to KPIC Grid")
            # ax.plot(model.wlen,model.flux)
            
            # # now plot the original model
            # ax = axes[0]
            # ax.set_title("Original Model")
            # mask = (m_w > min(model.wlen)) & (m_w < max(model.wlen)) 
            # ax.plot(m_w[mask], m_f[mask])
            
            # ax = axes[2]
            # ax.set_title("Data")
            # ax.plot(data_spec.wlen,data_spec.flux)
            # plt.show()
            # plt.close()
            
            model.metadata.update(sel)  # so that the dataset knows what fiber/night/order/etc this spectrum corresponds to
                        
            # plt.show()
            # plt.close() 
            crires_models.append(model)
            
        return Dataset(crires_models, dims=self.data.dims)