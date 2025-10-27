import logging
from typing import List
import numpy as np 
import pandas as pd
from os.path import join

import scipy.ndimage as ndi

import matplotlib.pyplot as plt

from .crires.crires_spectrum import CRIRESSpectrum
from .crires.plotting import plot_crires_model
from sika.implementations.spectroscopy.utils import optimize_scale_factors

from .spectra.spectrum import Spectrum
from sika.modeling import Sampler, Dataset

# this is required or pickling of things like lambdas will not work
import dill
import dynesty.utils
dynesty.utils.pickle_module = dill

from sika.utils import savefig

__all__ = ["NComponentSampler", "scale_model_to_order"]

# adapted from jerry xuan
def scale_model_to_order(order_wlen: np.ndarray, model_wlen: np.ndarray, model_flux:np.ndarray, filter_type, filter_size) -> np.ndarray:
    # take a model representing the entire spectrum and crop/scale it to a specific order in the data
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

class NComponentSampler(Sampler[CRIRESSpectrum, Spectrum]):
    """A :py:class:`~sika.modeling.sampler.Sampler` that fits N spectra to a spectroscopic dataset.

    :type Sampler: :py:class:`~sika.modeling.sampler.Sampler`
    """
    
    def _make_model(self) -> Dataset[CRIRESSpectrum]:
        # get individual spectra for each model component
        modeled_datasets = [m.make_model() for m in self.models]
        
        spectra = []
        for selector, data_spectrum in self.data:
            individual_model_fluxes = {m.name: [] for m in self.models}
            individual_model_wlens = {m.name: [] for m in self.models}
            modeled_spectra = [md.values(selector) for md in modeled_datasets]
            wlen, comb_flux, comb_errors, comb_residuals, comb_scale_factors, betas = [], [], [], [], [], []
            orderwise_model_fluxes = []
            i = 0
            for (o_wlen, o_flux, o_errors) in zip(data_spectrum.wlen, data_spectrum.flux, data_spectrum.errors):
            # for (o_wlen, o_flux, o_errors) in zip(data_spectrum.wlen_by_order, data_spectrum.flux_by_order, data_spectrum.error_by_order):
                assert len(o_wlen) == len(o_flux) == len(o_errors), f"Data wavelength ({len(o_wlen)}), flux ({len(o_flux)}), and error ({len(o_errors)}) arrays must be of the same length within an order"
                model_fluxes = [scale_model_to_order(o_wlen, s.wlen_flat,s.flux_flat, self.filter_type, self.filter_size) for s in modeled_spectra]
                bad_mask = np.logical_or.reduce([np.isnan(f) for f in model_fluxes])
                
                for (start_wlen, end_wlen) in self.masked_ranges:
                    bad_mask[(o_wlen >= start_wlen) & (o_wlen <= end_wlen)] = True
                
                model_fluxes = [f[~bad_mask] for f in model_fluxes]
                o_wlen = o_wlen[~bad_mask]
                o_flux = o_flux[~bad_mask]
                o_errors = o_errors[~bad_mask]
                
                for m, f in zip(self.models, model_fluxes):
                    individual_model_fluxes[m.name].append(f)
                    individual_model_wlens[m.name].append(o_wlen)
                
                scale_factors, beta = optimize_scale_factors(o_flux, o_errors, model_fluxes)
                # if np.any(scale_factors < 0):
                #     self.write_out(f"order {i} negative scale factors found.")
                #     self.write_out("saving self and exiting.")
                #     with open(join(self.outdir,"failed_sampler.pkl"),"wb") as f:
                #         dill.dump(self,f)
                #     raise ValueError("Negative scale factors found. See failed_sampler.pkl for details.")
                # self.write_out(f"order {i} scale factors:", scale_factors)
                i += 1
                combined_flux = sum(f * sf for f, sf in zip(model_fluxes, scale_factors))
                residuals = o_flux - combined_flux  # compute the residuals here instead of later just because its convenient. we'll store them in the metadata and pull them out later
                wlen.append(o_wlen)
                comb_flux.append(combined_flux)
                comb_errors.append(o_errors * beta)  # scale the errors by the beta factor
                comb_residuals.append(residuals)
                comb_scale_factors.append(scale_factors)
                betas.append(beta)
            
            # param_dict = {model.name: model.parameter_set.sel(selector) for model in self.models}
            if np.any(np.concatenate(comb_scale_factors) < 0):
                self.write_out("negative scale factors!", np.array(comb_scale_factors), "| data selector:", selector, "| parameters:", self.params)

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
                wlen=np.concatenate(wlen),
                flux=np.concatenate(comb_flux),
                errors = np.concatenate(comb_errors),
                metadata = { **selector, 
                            "scale_factors": comb_scale_factors, 
                            "n_comp_masked_ranges":self.masked_ranges, 
                            "beta": betas, 
                            "residuals": np.concatenate(comb_residuals), 
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
            nights = self.best_models.coords["night"]
            rows = []

            for n in nights:
                row = {"night":n}
                for k, v in self.param_w_uncert.items():
                    if "rv" in k and n not in k:
                        continue
                    pname = k.split(" (night=")[0].replace(": ","_")
                    val, (val_plus, val_minus,val_std) = v
                    
                    row[pname] = val
                    row[pname+"_plus"] = val_plus
                    row[pname+"_minus"] = val_minus
                    row[pname+"_std"] = val_std
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df = df.sort_values("night")
            df.to_csv(join(self.outdir,"best_params.csv"),index=None)
        except Exception as e:
            self.write_out(f"Error saving best_params.csv: {e}. continuing anyway", level=logging.ERROR)

    def visualize_results(self):
        show = self.config.get("show_plots", False)
        try:    
            fig, axes = plt.subplots(
                nrows=self.data.size, ncols=1, figsize=(18, 6*self.data.size)
            )
            try:
                [a for a in axes]
            except:
                axes = [axes]
            for (selector, data_spectrum), ax in zip(self.data, axes):
                model_spectrum = self.best_models.values(selector)
                for o_wlen, o_flux in zip(data_spectrum.wlen, data_spectrum.flux):
                    ax.plot(o_wlen, o_flux, color="gray", label="Data")
                    ax.plot(o_wlen, np.interp(o_wlen, model_spectrum.wlen, model_spectrum.flux), color="red", label="Model",alpha=0.75)
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
                for order in range(data_spectrum.norders):
                    plot_crires_model(model_spectrum, data_spectrum, order, selector)
                    savefig(f"best_o{order}_{selector}.png".replace("{",'').replace("}",'').replace('\'','').replace(": ","_"), config=self.config, outdir=self.outdir)
                    plt.close()
        except Exception as e:
            self.write_out('Order-by-order best fit plots failed',level=logging.WARNING)
            print(e)
            
            
        plt.cla()
        
        super().visualize_results()

            
