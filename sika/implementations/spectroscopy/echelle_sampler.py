import logging
from typing import List, Optional, Union
import numpy as np 
import pandas as pd
from os.path import join

import scipy.ndimage as ndi

import matplotlib.pyplot as plt

from .crires.crires_spectrum import CRIRESSpectrum
from .spectra.plotting import plot_model_v_data
from sika.implementations.spectroscopy.utils import optimize_scale_factors, ErrorInflationParameterSet
from sika.modeling.parameter_set import joint_iter as joint_iter_paramset
from .spectra.spectrum import Spectrum
from sika.modeling import Sampler, Dataset, ConstraintViolation, AuxiliaryParameterSet, Parameter, ParameterSet, PriorTransform

# this is required or pickling of things like lambdas will not work
import dill
import dynesty.utils
dynesty.utils.pickle_module = dill

from sika.utils import savefig, format_selector_string

__all__ = ["NComponentEchelleSampler", "scale_model_to_order"]

# adapted from jerry xuan
def scale_model_to_order(order_wlen: np.ndarray, model_wlen: np.ndarray, model_flux:np.ndarray, filter_type, filter_size) -> np.ndarray:
    # take a model representing the entire spectrum and crop/scale it to a specific order in the data
    # performs continuum subtraction with a filter of type 'filter_type' and of size 'filter_size'
    # returns the scaled flux array for that order
    f = np.interp(order_wlen, model_wlen, model_flux)
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
    # print("----- done scaling -------")
    return f


class NComponentEchelleSampler(Sampler[Spectrum, Spectrum]):
    """A :py:class:`~sika.modeling.sampler.Sampler` that fits N spectra to a spectroscopic dataset.

    :type Sampler: :py:class:`~sika.modeling.sampler.Sampler`
    """
    
    def __init__(self, *args, flux_scale_parameters:Optional[AuxiliaryParameterSet]=None, error_inflation_terms:Optional[ErrorInflationParameterSet]=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimize_flux_scale = True
        if flux_scale_parameters is not None:
            # assert error_inflation_terms is not None, "If flux scaling are fit parameters, error_inflation_terms parameters must also be provided."
            self.optimize_flux_scale = False
            self.flux_scale_parameters = flux_scale_parameters  # make this a property so that we can keep track of which set of parameters are the flux scale parameters
            self.aux_param_sets.append(self.flux_scale_parameters)  # add this to the list of auxiliary parameter sets so that it will be fit by the sampler
            self.error_inflation_terms = error_inflation_terms
            if error_inflation_terms is not None:
                self.aux_param_sets.append(self.error_inflation_terms)
            

    def _make_model(self) -> Dataset[Spectrum]:
        # get individual spectra for each model component
        modeled_datasets = [m.make_model() for m in self.models]
        
        spectra = []
        for selector, data_spectrum in self.data:
            individual_model_fluxes = {}
            individual_model_wlens = {}
            # individual_model_fluxes = {m.name: [] for m in self.models}
            # individual_model_wlens = {m.name: [] for m in self.models}
            modeled_spectra = [md.values(selector) for md in modeled_datasets]
            # wlen, comb_flux, comb_errors, comb_residuals, comb_scale_factors, betas = [], [], [], [], [], []
            # orderwise_model_fluxes = []
            
            d_wlen, d_flux, d_errors = np.array(data_spectrum.wlen,copy=True), np.array(data_spectrum.flux,copy=True), np.array(data_spectrum.errors,copy=True)
            assert len(d_wlen) == len(d_flux) == len(d_errors), f"Data wavelength ({len(d_wlen)}), flux ({len(d_flux)}), and error ({len(d_errors)}) arrays must be of the same length!"
            model_fluxes = [scale_model_to_order(d_wlen, np.array(s.wlen,copy=True), np.array(s.flux,copy=True), self.filter_type, self.filter_size) for s in modeled_spectra]
            bad_mask = np.logical_or.reduce([np.isnan(f) for f in model_fluxes])
            
            for (start_wlen, end_wlen) in self.masked_ranges:
                bad_mask[(d_wlen >= start_wlen) & (d_wlen <= end_wlen)] = True
            
            model_fluxes = [f[~bad_mask] for f in model_fluxes]
            d_wlen = d_wlen[~bad_mask]
            d_flux = d_flux[~bad_mask]
            d_errors = d_errors[~bad_mask]
            
            for m, f in zip(self.models, model_fluxes):
                # individual_model_fluxes[m.name].append(f)
                individual_model_fluxes[m.name] = f
                # individual_model_wlens[m.name].append(d_wlen)
                individual_model_wlens[m.name] = d_wlen
            
            if self.optimize_flux_scale:
                scale_factors, beta = optimize_scale_factors(d_flux, d_errors, model_fluxes)
            else:
                beta = 1  # not doing error inflation 
                if self.error_inflation_terms is not None:
                    beta = self.error_inflation_terms.beta.values(selector)
                    
                # aggregate the scale factors by pulling them out of the user-provided parameter set (which gets fit by the sampler)
                scale_params = self.flux_scale_parameters.sel(selector)
                scale_factors = []
                for p in scale_params.values():
                    scale_factors.append(p)
                assert len(scale_factors) == len(model_fluxes), f"Scale parameters are not the correct shape! We have {len(model_fluxes)} models but {len(scale_factors)} scale factors: {scale_params}"
                
            combined_flux = sum(f * sf for f, sf in zip(model_fluxes, scale_factors))
            residuals = d_flux - combined_flux  # compute the residuals here instead of later just because its convenient. we'll store them in the metadata and pull them out later
            errors = d_errors * beta
            
            # betas.append(beta)
            # wlen.append(d_wlen)
            # comb_flux.append(combined_flux)
            # comb_errors.append(d_errors * beta)  # scale the errors by the beta factor
            # comb_scale_factors.append(scale_factors)
            # comb_residuals.append(residuals)
            
            # param_dict = {model.name: model.parameter_set.sel(selector) for model in self.models}
            if np.any(scale_factors == 0):
                # raise ConstraintViolation("Scale factors of zero!")
                self.write_out("scale factors of zero!")
                # self.write_out("scale factors of zero!", np.array(comb_scale_factors), "| data selector:", selector, "| parameters:", self.params)

            model_info = {}
            for model, spec in zip(self.models, modeled_spectra):
                model_info[model.name] = {
                    "classname":model.__class__.__name__,
                    "dispname":model.display_name,
                    "metadata": spec.metadata,
                    "flux": individual_model_fluxes[model.name],
                    "wlen": individual_model_wlens[model.name],
                }
            
            s = Spectrum(
                parameters={ },
                wlen=d_wlen,
                flux=combined_flux,
                errors = errors,
                metadata = { **selector, 
                            "scale_factors": scale_factors, 
                            "n_comp_masked_ranges":self.masked_ranges, 
                            "beta": beta, 
                            "residuals": residuals, 
                            "dataset_meta": data_spectrum.metadata, 
                            "dataset_params":data_spectrum.parameters, 
                            "n_free_params":self.nparams,
                            "models": model_info
                        }
            )
            spectra.append(s)
                
        ds = Dataset(spectra, dims=self.data.dims)
        return ds
    
    def get_errors_and_residuals(self, modeled_ds: Dataset[Spectrum]):
        """ :meta private: """
        all_errors, all_residuals = [], []
        for _, modeled_spec in modeled_ds:
            all_errors.append(modeled_spec.errors)
            all_residuals.append(modeled_spec.metadata["residuals"])

        errors = np.concatenate(all_errors)
        residuals = np.concatenate(all_residuals)
        return errors, residuals
    
    def _setup(self):
        target_cfg = self.config[self.config["target"]]
        data_cfg = target_cfg["data"]
        self.filter_size = data_cfg["filter_size"]
        self.filter_type = data_cfg["filter_type"]
        self.bp_sigma = data_cfg["bp_sigma"]
        
        self.masked_ranges = target_cfg.get("n_comp",{}).get("masked_ranges",[])  # ranges of wlen that we will mask
        
        super()._setup()
        
        self.write_out("Setup complete.")
                
    
    def save_results(self):
        super().save_results()
        
        try:
            uncert2 = {}
            for k, (val, uncerts) in self.param_w_uncert.items():
                uncert2[k] = dill.dumps(([val]+list(uncerts)))
            rows = []
            self.set_model_params(list(uncert2.values()))
            for sel, pval in joint_iter_paramset(*self.params):
                row = sel
                pval = [dill.loads(p) for p in pval]
                for pname, v in zip(self.short_param_names, pval):
                    val, val_plus, val_minus,val_std = v     
                    row[pname] = val
                    row[pname+"_plus"] = val_plus
                    row[pname+"_minus"] = val_minus
                    row[pname+"_std"] = val_std
                rows.append(row)
            df = pd.DataFrame(rows)
            try:
                df = df.sort_values("night")
            except:
                pass
            df.to_csv(join(self.outdir,"best_params.csv"),index=None)
        except Exception as e:
            self.write_out(f"Error saving best_params.csv: {e}. continuing anyway", level=logging.ERROR)

    def visualize_results(self):
        show = self.config.get("show_plots", False)
        try:    
            fig, axes = plt.subplots(
                nrows=self.data.size, ncols=1, figsize=(18, 6*self.data.size)
            )
            axes = np.atleast_1d(axes)
            # try:
            #     [a for a in axes]
            # except:
            #     axes = [axes]
            for (selector, data_spectrum), ax in zip(self.data, axes):
                model_spectrum = self.best_models.values(selector)
                d_wlen, d_flux = data_spectrum.wlen, data_spectrum.flux
                ax.plot(d_wlen, d_flux, color="gray", label="Data")
                ax.plot(d_wlen, np.interp(d_wlen, model_spectrum.wlen, model_spectrum.flux), color="red", label="Model",alpha=0.75)
                ax.set_title(", ".join(f"{k} = {v}" for k, v in selector.items()))
                ax.legend()
            plt.suptitle("Best Fit Model vs Data")
            plt.tight_layout()
            savefig("best_fit_model.png", config=self.config, outdir=self.outdir)
            if show:
                plt.show()
        except Exception as e:
            self.write_out('Combined best model plot failed',level=logging.WARNING)
            print(e)
                        
        try:
            n_nights = len(self.best_models.selectors)
            fig, axes = plt.subplots(nrows=n_nights, ncols=1,figsize=(6,3*n_nights))
            try:
                [a for a in axes]
            except:
                axes = [axes]
            for (sel, model), ax in zip(self.best_models,axes):
                ax.set_title(sel["night"])
                scale_factors = np.array(model.metadata["scale_factors"])
                model_names = [minfo["dispname"] for minfo in model.metadata["models"].values()]
                for i, component in enumerate(model_names):
                    s_f = scale_factors[:,i]
                    ax.plot(np.arange(len(s_f)),s_f,label=component)
                ax.legend()
                ax.set_ylabel("Scale Factor")
                ax.set_xlabel("Order")
            plt.tight_layout()
            savefig("scale_factors.png", self.config, outdir=self.outdir)
        except Exception as e:
            self.write_out('Scale-factor plot failed',level=logging.WARNING)
            print(e)
            
        plt.cla()
        try:
            for selector, data_spectrum in self.data:
                model_spectrum = self.best_models.values(selector)
                plot_model_v_data(model_spectrum, data_spectrum, selector)
                savefig(f"best_{format_selector_string(selector)}.png", config=self.config, outdir=self.outdir)
                plt.close()
        except Exception as e:
            self.write_out('Order-by-order best fit plots failed',level=logging.WARNING)
            print(e)
            
            
        plt.cla()
        
        super().visualize_results()

            
