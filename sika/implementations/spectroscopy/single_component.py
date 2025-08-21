import sys
from os import makedirs
from os.path import join
from typing import List, Callable, Union, Optional, Dict
import numpy as np 
import dynesty.plotting as dyplot
import itertools
from dynesty import NestedSampler
# from schwimmbad import MPIPool

import matplotlib.pyplot as plt
from logging import Logger

from sika.modeling.priors import PriorTransform

from .spectra import Spectrum
from sika.provider import Provider
from sika.task import Task
from sika.modeling import Model, Dataset, ParameterSet, DataLoader, LnLikelihood

from sika.config import Config
from sika.utils import save_bestfit_dict, savefig, plot_corner

class SingleComponentModel(Task):
    supported_samplers = ['dynesty']
    
    def __init__(self, outdir, data_provider: DataLoader[Spectrum], model: Model[Spectrum], *args, loss=LnLikelihood(), data_params:Optional[Dict]=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.outdir = outdir
        makedirs(outdir, exist_ok=True)
        self.data_provider = data_provider
        self.model = model
        self.loss = loss
        self.sampler = None
        self.data = None
        self._data_params = data_params if data_params else {}
        self._data_selectors = []

    @property
    def previous(self):
        p = [self.model]
        if self.data_provider is not None:
            p.append(self.data_provider)
        return p
        
    # def configure(self, config:Union[None,Config], logger: Union[None,Logger]):
    #     self.data_provider.configure(config, logger)
    #     self.model.configure(config, logger)
    #     super().configure(config, logger)

    @property 
    def coords(self):
        return self.model.coords
        # because we only have one model our coords are just the model coords
    
    @property
    def dims(self):
        return self.model.dims
    
    @property
    def nparams(self):
        return self.model.parameter_set.nvals
    
    def explain_shape(self) -> str:
        return f"{self.model.__class__.__name__}:\n{self.model.parameter_set.shape}"
    
    def _setup(self):
        target_cfg = self.config[self.config["target"]]
        sampler_type = target_cfg["sampler_type"]
        if sampler_type not in self.supported_samplers:
            raise ValueError(f"Sampler type {sampler_type} not supported. Supported samplers: {self.supported_samplers}")
        self.write_out("Loading data from provider...")
        self.data = self.data_provider(self._data_params)  # get the data that we are fitting to
        self.write_out("Loaded data.")
        data_coords = self.data.coords
        self.write_out("Setting up model with data coordinates:", data_coords)
        self.model.set_coords(data_coords)  # inform the model of the data coordinates (will inform parameters)
        self.write_out("Model setup complete. Model coordinates:", self.model.coords)
        self.write_out("Preprocessing data...")
        self.data = self.preprocess(self.data)
        self.write_out("Data preprocessing complete.")
        

    def make_model(self,parameters:List[float]) -> Dataset[Spectrum]:
        modeled = self.model(parameters)
        processed = self.preprocess(modeled)
        return processed
    
    def iterate(self, parameters: List[float]) -> float:
        """
        Generate a model spectrum for the single component using the provided parameters.
        """
        modeled_dataset = self.make_model(parameters) # this is a dataset of model spectra
        loss_val = 0
        for selector in self._data_selectors:
            # select the data spectrum for this selector
            data_spectrum = self.data.values(selector)
            if not isinstance(data_spectrum, Spectrum):
                raise ValueError(f"Data spectrum for selector {selector} is not a Spectrum, got {type(data_spectrum)}")
            
            # select the model spectrum for this selector
            model_spectrum = modeled_dataset.values(selector)
            if not isinstance(model_spectrum, Spectrum):
                raise ValueError(f"Model spectrum for selector {selector} is not a Spectrum, got {type(model_spectrum)}")
            
            # calculate the residual and loss
            resid = data_spectrum.flux - model_spectrum.flux
            calc_loss = self.loss(data_spectrum.errors, resid)
            # self.write_out(f"Loss for selector {selector}: {calc_loss}")
            loss_val += calc_loss

        return loss_val

    def prior_transform(self,u:np.ndarray) -> np.ndarray:
        x = np.copy(u)
        for i, param in enumerate(self.model.prior_transforms()):
            x[i] = param.prior_transform(u[i])
        return x
        
    # no-op preprocess
    def preprocess(self, spectrum: Dataset[Spectrum]) -> Dataset[Spectrum]:
        return spectrum
        
    def fit(self):
        self.write_out("Fitting model with parameters:", self.model.parameter_set.unfrozen)
        
        target_cfg = self.config[self.config["target"]]
        sampler_type = target_cfg["sampler_type"]
        
        if sampler_type == 'dynesty':
            dynesty_cfg = target_cfg["dynesty"]
            num_live = dynesty_cfg["nlive"]
            num_walks = dynesty_cfg["nwalks"]
            bound_method = dynesty_cfg["bound_method"]
            sample_method = dynesty_cfg["sample_method"]
            dlogz_stop_crit = dynesty_cfg["dlogz_stop_crit"]
            live_file = join(self.outdir,"dynesty_live.pkl")
            
            self.write_out("Starting sampling with Dynesty")
            sampler = NestedSampler(self.iterate, self.prior_transform, self.nparams, nlive=num_live, 
                                    bound=bound_method, sample=sample_method, walks=num_walks) # pool=mypool,
            sampler.run_nested(checkpoint_file=live_file, dlogz=dlogz_stop_crit)
            self.write_out("Sampling complete, gathering results")
            
            res = sampler.results
            # get best fit
            logprob_chain = res['logl']

            # new dynesty 2.1.1
            max_params = res['samples'][np.argmax(logprob_chain)]
            plot_chain = res.samples_equal()
            
            self.model.parameter_set.set_values_flat(max_params)
            

            param_names = [p.name for p in self.model.parameter_set.unfrozen]
            best_fit_dict = dict(zip(param_names, max_params))
            print("Best fit parameters:", best_fit_dict)
            save_bestfit_dict(best_fit_dict, join(self.outdir,"best_fit_params.pkl"))
            
            try:
                fig, axes = dyplot.traceplot(res, labels=param_names)
                savefig("traceplots.png", config=self.config, outdir=self.outdir)
                plt.show()
                plt.close()
                plot_corner(plot_chain, param_names, fs=12, fs2=10)  # fs=fontsize
                savefig("corner_plot.png", self.config, outdir=self.outdir)
                plt.show()
            except Exception as e:
                print('Corner / trace plot failed')
                print(e)

            return best_fit_dict