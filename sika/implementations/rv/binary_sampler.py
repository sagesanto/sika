import os
import shutil
from os.path import join, exists
import pandas as pd
from typing import Union, List, Optional
from dataclasses import dataclass
from astropy import units as u
import logging
from sika.config import Config, config_path, configure_logger
from sika.modeling import (
    Model,
    Parameter,
    ParameterSet,
    RelativeParameter,
    PriorTransform,
    Uniform,
    Normal,
    NullPriorTransform,
    Dataset,
    Sampler
)
from sika.product import Product, DFProduct
from sika.modeling.params import joint_iter
from sika.utils import (
    sika_argparser,
    parse_sika_args,
    parse_path,
    write_out as _write_out,
    get_pool,
)

from sika.implementations.rv.RV import RV, BinaryRV
from sika.implementations.rv.kepler import keplerian_function
import numpy as np
import matplotlib.pyplot as plt

# assumes that data is concatenated so that the first component's rv is first and then the second component's rv is second
class BinaryRVSampler(Sampler[BinaryRV, BinaryRV]):
    def _make_model(self) -> Dataset[BinaryRV]:
        modeled_rvs = [m.make_model() for m in self.models]
        all_rvs = []    
        for selector, data in self.data:
            matching: List[BinaryRV] = [md.values(selector) for md in modeled_rvs]
            for rv in matching:
                rv.metadata.update(selector)
                all_rvs.append(rv)
        return Dataset(all_rvs, dims=self.data.dims)
    
    def get_errors_and_residuals(self, modeled_ds: Dataset[BinaryRV]):
        all_errors, all_residuals = [], []
        for sel, data in self.data:
            modeled_rv = modeled_ds.values(sel)
            jitter = self.aux_param_sets[0].jitter.values(sel)
            rv_jitter = np.ones_like(data.rv1.rv) * jitter
            err_1 = np.sqrt(data.rv1.rv_err**2 + rv_jitter**2)
            err_2 = np.sqrt(data.rv2.rv_err**2 + rv_jitter**2)
            all_errors.append(err_1)
            all_residuals.append(modeled_rv.rv1.rv - data.rv1.rv)
            all_errors.append(err_2)
            all_residuals.append(modeled_rv.rv2.rv - data.rv2.rv)
        errors = np.concatenate(all_errors)
        residuals = np.concatenate(all_residuals)
        return errors, residuals

    def visualize_results(self):
        show = self.config.get("show_plots", False)

        fig, axes = plt.subplots(nrows=self.data.size, ncols=1, figsize=(18, 6*self.data.size))
        try:
            [a for a in axes]
        except:
            axes = [axes]
        try:
            
            for (selector, data), ax in zip(self.data, axes):
                    model_rvs = self.best_models.values(selector)
                    t1, t2 = data.rv1.t, data.rv2.t
                    rv1, rv2 = data.rv1.rv, data.rv2.rv
                    t0 = min(np.concatenate((t1, t2)))
                    tf = max(np.concatenate((t1, t2)))
                    model_t = np.linspace(t0, tf, 1000)
                    amplitude_1 = model_rvs.parameters["amplitude_1"]
                    amplitude_2 = model_rvs.parameters["amplitude_2"]
                    period = model_rvs.parameters["period"]
                    eccentricity = model_rvs.parameters["eccentricity"]
                    omega_planet = model_rvs.parameters["omega_planet"]
                    tau = model_rvs.parameters["tau"]
                    offset = model_rvs.parameters["offset"]
                    drift = model_rvs.parameters["drift"]
                    model_rv1 = keplerian_function(model_t, amplitude_1, period, eccentricity, omega_planet, tau, offset) + drift * (model_t - t0)
                    model_rv2 = keplerian_function(model_t, amplitude_2, period, eccentricity, omega_planet + np.pi, tau, offset) + drift * (model_t - t0)
                    ax.errorbar(t1/86400, rv1/1e3, yerr=data.rv1.rv_err/1e3, fmt='o', label='Data 1', color='tab:blue')
                    ax.errorbar(t2/86400, rv2/1e3, yerr=data.rv2.rv_err/1e3, fmt='o', label='Data 2', color='tab:orange')
                    ax.plot(model_t/86400, model_rv1/1e3, label='Model 1', color='tab:blue')
                    ax.plot(model_t/86400, model_rv2/1e3, label='Model 2', color='tab:orange')
                    ax.set_title(", ".join(f"{k} = {v}" for k, v in selector.items()))
                    ax.legend(ncols=4,frameon=False,fontsize=8)
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Radial Velocity")
                    ax.set_xlabel("t - t$_0$ (days)")
                    ax.set_ylabel("Radial Velocity (km/s)")
                    ax.plot(model_t/86400, (model_t * drift + offset)/1e3, linestyle="--", color="gray")
                    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
            plt.savefig(join(self.outdir, "rv_fit.png"), dpi=300, bbox_inches='tight')
            if show:
                plt.show()
            plt.close(fig)
        except Exception as e:
            self.write_out(f"Error making RV plot: {e}", level=logging.ERROR)
        
        super().visualize_results()