import sys
import os
from os import makedirs
from os.path import join, exists
import pickle
import logging
from typing import Any, List, Union, Optional, Dict, TypeVar, Generic, Tuple, Callable
import numpy as np 
import dynesty.plotting as dyplot
from dynesty import NestedSampler
import pandas as pd
import json
from abc import ABC, abstractmethod
from enum import Enum
import matplotlib.pyplot as plt

from sika.task import Task
from sika.modeling import Model, Dataset, DataLoader, LnLikelihood, Constraint, ConstraintViolation, AuxiliaryParameterSet, PriorTransform, ParameterSet


# this is required or pickling of things like lambdas will not work
import dill
import dynesty.utils
dynesty.utils.pickle_module = dill

import emcee

logging.getLogger('matplotlib').setLevel(logging.WARNING)  # suppress matplotlib debug messages
logging.getLogger('PIL').setLevel(logging.WARNING)  # suppress buggy tk PIL output

from sika.config import Config
from sika.utils import NodeSpec, NodeShape, save_bestfit_dict, savefig, plot_corner, get_mpi_info, get_process_info, get_pool, plot_chains_vs_priors
from sika.product import Product
from scipy.optimize import minimize

__all__ = ["Sampler"]

_WORKER_SAMPLER: Optional["Sampler[Any, Any]"] = None

class EmceeGuessType(Enum):
    Manual = "manual"
    Random = 'random'
    Minimize = 'minimize'
    # Optimize = 'optimize'  # should just remove this option bc its a blind search and not very reliable

def _init_worker_sampler(sampler: "Sampler[Any, Any]") -> None:
    global _WORKER_SAMPLER
    _WORKER_SAMPLER = sampler


def _worker_iterate(parameters: np.ndarray) -> float:
    if _WORKER_SAMPLER is None:
        raise RuntimeError("Worker sampler has not been initialized")
    return _WORKER_SAMPLER.iterate(parameters)


def _worker_prior_transform(unit_cube: np.ndarray) -> np.ndarray:
    if _WORKER_SAMPLER is None:
        raise RuntimeError("Worker sampler has not been initialized")
    return _WORKER_SAMPLER.prior_transform(unit_cube)


def _worker_log_posterior(theta: np.ndarray) -> float:
    if _WORKER_SAMPLER is None:
        raise RuntimeError("Worker sampler has not been initialized")
    return _WORKER_SAMPLER.log_posterior(theta)

D = TypeVar('D', bound=Product, covariant=True)  # type of the data Product
M = TypeVar('M', bound=Product, covariant=True)  # type of the model Product
class Sampler(Generic[D,M], Task, ABC):
    """ Base class for samplers that fit models to data using a likelihood function. """
    
    supported_samplers = ['dynesty', "pymultinest", "emcee"]

    def __init__(self, run_prefix:str, outdir, data: Union[Dataset[D], DataLoader[D]], models: List[Model[M]], *args, loss=LnLikelihood(), data_params: Optional[Dict] = None, restore_from: Optional[str]=None, constraints: Optional[List[Constraint]]=None, aux_params:Optional[AuxiliaryParameterSet]=None, **kwargs):
        """Initialize the Sampler.

        :param run_prefix: a name for the run, used to name output files
        :type run_prefix: str
        :param outdir: the directory to write output files to
        :type outdir: str
        :param data: the data to fit the models to, either a Dataset or a DataLoader that returns a Dataset
        :type data: Union[Dataset[D], DataLoader[D]]
        :param models: the models to fit to the data
        :type models: List[Model[M]]
        :param loss: loss function, defaults to LnLikelihood()
        :type loss: Loss, optional
        :param data_params: parameters to pass to a provided DataLoader, defaults to None
        :type data_params: Optional[Dict], optional
        :param restore_from: directory or filepath to resume sampling from. If sampling with dynesty, this should point to a ``dynesty_live.pkl`` file. If sampling with PyMultiNest, this should be the previous run's output directory joined with its ``run_prefix``, defaults to None
        :type restore_from: Optional[str], optional
        :param constraints: constraints that should be checked at every iteration. If the constraints fail, the loss function will report ``-np.inf`` as the loss. defaults to None
        :type constraints: Optional[List[Constraint]], optional
        :param aux_params: a set of auxiliary parameters that will be fit by the sampler but do not belong to a model. effective use requires subclassing Sampler and making use of the auxiliary params in that implementation. defaults to None
        :type aux_params: Optional[AuxiliaryParameterSet], optional
        """
        super().__init__(*args, **kwargs)
        if aux_params is None:
            aux_params = AuxiliaryParameterSet()
        self.aux_param_sets = [aux_params]   # sets of auxiliary parameters
        self.outdir = outdir
        self.run_prefix = run_prefix
        makedirs(outdir, exist_ok=True)
        self.models = models
        self.loss = loss
        self.sampler = None
        self.restore_from = restore_from
        
        self.mcmc_starting_guess = None
        
        self._data_params = data_params if data_params else {}
        self.loss_adjustments = []
        if isinstance(data, Dataset):
            self.data = data
            self.data_provider = None
        else:  # assume it's a DataLoader, will call later
            self.data_provider = data
            self.data: Dataset[D] = None
        self.constraints = constraints if constraints is not None else []

        # self.params = [p for model in models for p in model.params] + [p for pset in self.aux_param_sets for p in pset.unfrozen]
    
    @property
    def params(self):
        """A list of the unfrozen parameters of each model and auxiliary parameter set"""
        return [p for model in self.models for p in model.params] + [p for pset in self.aux_param_sets for p in pset.unfrozen]
    
    @property
    def previous(self):
        if self.data_provider is not None:
            return self.models + [self.data_provider]
        return self.models
        
    @property
    def param_names(self):
        """A list of the names of the unfrozen parameters of each model and auxiliary parameter set, including coordinate variants"""
        n = []
        for model in self.models:
            n.extend(model.param_names)
        for pset in self.aux_param_sets:
            n.extend(pset.all_names())
        return n
    
    @property
    def short_param_names(self):
        """A list of the names of the unfrozen parameters of each model and auxiliary parameter set, excluding coordinate variants"""
        n = []
        for model in self.models:
            n.extend(model.short_param_names)
        for pset in self.aux_param_sets:
            n.extend(pset.short_names())
        return n
    
    @property
    def parameter_sets(self) -> List[ParameterSet]:
        """A list containing the parameter sets of each model, plus each auxiliary parameter set"""
        psets = []
        for m in self.models:
            psets.extend(m.parameter_sets)
        psets.extend(self.aux_param_sets)
        return psets

    @property
    def flattened_guess(self) -> np.ndarray | None:
        guesses = []
        for pset in self.parameter_sets:
            flattened = pset.flattened_guess()
            if flattened is None:
                return None
            if len(flattened):
                guesses.append(flattened)
        if not guesses:
            return np.array([])
        return np.concatenate(guesses)
        
    def node_spec(self) -> NodeSpec:
        return NodeSpec(
            shape=NodeShape.RECT,
            label=self.__class__.__name__,
            color="#FFCC00",
            ID=self.ID,
        )
        
    def args_to_dict(self):
        args = {}
        args["outdir"] = self.outdir
        args["data_provider"] = self.data_provider
        args["models"] = self.models
        args["loss"] = self.loss
        args["loss_adjustments"] = [description for _,description in self.loss_adjustments]
        args["data_params"] = self._data_params
        args["restore_from"] = self.restore_from
        args["constraints"] = self.constraints
        args["aux_params"] = self.aux_param_sets
        args["mcmc_starting_guess"] = self.mcmc_starting_guess
        return args
        
    # def configure(self, config:Union[None,Config], logger: Union[None,Logger]):
    #     # in case the data provider is a Provider, configure it
    #     if self.data_provider is not None:
    #         self.data_provider.configure(config, logger)
    #     for model in self.models:
    #         model.configure(config, logger)
    #     super().configure(config, logger)
    
    def _group_params(self, parameters:List[float]):
        grouped_params = []
        running_param_index = 0
        for nparams in self.nparams_per:
            params = parameters[running_param_index:running_param_index+nparams]
            running_param_index += nparams
            grouped_params.append(params)
        return grouped_params
    
    def set_model_params(self,parameters:List[float]):
        """ :meta private: """
        grouped_params = self._group_params(parameters)
        aux_psets = []
        for _ in self.aux_param_sets:
            aux_psets.append(grouped_params.pop())
        for params, model in zip(grouped_params, self.models):
            model.set_params(params)
        aux_psets.reverse()
        for i, pset in enumerate(aux_psets):
            self.aux_param_sets[i].set_values_flat(np.array(pset))
    
    def make_model(self,parameters:List[float]) -> Dataset[M]:
        """ Make a combined model from the flattened parameters and return it. """
        # divvy up the homogenous array of parameters and set the param values of each model
        self.set_model_params(parameters)
        
        # raises on failure, which is then caught in iterate
        for constraint in self.constraints:
            constraint.validate(raise_on_invalid=True)
        return self._make_model()
    
    @abstractmethod
    def _make_model(self) -> Dataset[M]:
        """Make the combined model for evaluation using the Sampler's models, whose parameters have **already been set** 

        :return: the model to be compared to the data
        :rtype: Dataset[M]
        """
    
    @abstractmethod
    def get_errors_and_residuals(self, modeled_ds: Dataset[M]) -> Tuple[np.ndarray, np.ndarray]:
        """Take the just-created Dataset of modeled products ``modeled_ds`` and return errors and residuals (presumably by comparing modeled_ds and self.data) for use in the log likelihood calculation 

        :param modeled_ds: the just-created Dataset of models to evaluate 
        :type modeled_ds: Dataset[M]
        :return: errors, residuals, each a 1d np array
        :rtype: np.ndarray, np.ndarray
        """
    
    def iterate(self, parameters: List[float]) -> float:
        """
        Make a combined model spectrum from the parameters and calculate the loss.
        This function is repeatedly called by the sampler during fitting.
        """
        try:
            modeled_ds = self.make_model(parameters)
        except ConstraintViolation as e:
            self.write_out(f"Constraint(s) violated for parameters {parameters}: {e}. Returning -np.inf")
            return -np.inf
        errors, residuals = self.get_errors_and_residuals(modeled_ds)
        loss = self.loss(errors, residuals)
        # (loss: float, parameters: List[float], model: Dataset, data: Dataset, config: Config)
        for adjustment, description in self.loss_adjustments:
            adj_value = adjustment(loss, parameters, modeled_ds, errors, residuals, self.data, self.config)
            loss += adj_value
            self.write_out(f"Applied loss adjustment '{description}': {adj_value}, new loss: {loss}", level=logging.DEBUG)
        if np.isnan(loss) or np.isinf(loss):
            self.write_out(f"WARNING: loss is {loss} for parameters {parameters}", level=logging.WARNING)
        return loss

    def prior_transform(self, u:np.ndarray) -> np.ndarray:
        """Transform an array of unit cube variables to the parameter-space by applying each parameter's prior transform.

        :param u: the unit array, provided by a sampler like PyMultiNest or dynesty 
        :type u: np.ndarray
        :return: the live points, transformed to parameter-space
        :rtype: np.ndarray
        """
        
        x = np.copy(u)
        i = 0
        for m in self.models:
            for transform in m.prior_transforms():
                x[i] = transform(u[i])
                i += 1
        for pset in self.aux_param_sets:
            for transform in pset.get_unfrozen_transforms():
                x[i] = transform(u[i])
                i += 1
        return x
    
    def prior_transforms(self) -> List[PriorTransform]:
        """ The :py:class:`~sika.modeling.PriorTransform` objects for each unfrozen parameter"""
        p = [t for m in self.models for t in m.prior_transforms()]
        for pset in self.aux_param_sets:
            p.extend(pset.get_unfrozen_transforms())
        return p
    
    def _setup(self):
        """ :meta private: """
        self.target_cfg = self.config[self.config["target"]]
        
        self.sampler_type = self.target_cfg["sampler_type"]
        if self.sampler_type not in self.supported_samplers:
            raise ValueError(f"Sampler type {self.sampler_type} not supported. Supported samplers: {self.supported_samplers}")
        
        if self.data_provider is not None: # we were passed a DataLoader instead of a dataset
            self.write_out("Loading data from provider...")
            self.data = self.data_provider(
                self._data_params
            )  # get the data that we are fitting to
            self.write_out("Loaded data.")
        data_coords = self.data.coords
        self.write_out("Setting up models with data coordinates:", data_coords)
        for model in self.models:
            model.set_coords(
                data_coords
            )  # inform the model of the data coordinates (will inform parameters)
        for pset in self.aux_param_sets:
            pset.set_coords(data_coords)
        self.nparams_per = [model.nvals for model in self.models] + [auxpset.nvals for auxpset in self.aux_param_sets]
        self.write_out("Number of parameters per model:", self.nparams_per[:-len(self.aux_param_sets)])
        self.write_out("Number of auxiliary parameters:", self.nparams_per[-len(self.aux_param_sets)])
        self.nparams = sum(self.nparams_per)
        self.write_out("Model setup complete.")
        if self.data.coords:
            self.write_out("Model coordinates:")
            for m in self.models:
                self.write_out(f"  {m.name}: ")
                for k, v in m.coords.items():
                    self.write_out(f"    {k}: {v}")
            self.write_out("Model shapes:")
            for m in self.models:
                self.write_out(f"  {m.name}: ")
                self.write_out(m.explain_shape())

        self.write_out("Parameters per model:", self.nparams_per)
        self.write_out("Total number of parameters:", self.nparams)
        
        self.write_out("Writing model architecture to file...")
        with open(join(self.outdir, "model.json"), "w+") as f:
            f.write(self.json())
            
        self.write_out("Writing parameter names to file...")
        with open(join(self.outdir,'parameter_names.txt'), 'w+') as f:
            f.write('\n'.join(self.param_names))

    def add_loss_adjustment(self, adjustment: Callable[[float, List[float], Dataset, Dataset, np.ndarray, np.ndarray, Config], float], description: str):
        """Add a callable that takes the current loss, parameters, model, data, and config as input and returns a loss adjustment. Will be called at every iteration of the model and directly added to the loss

        :param adjustment: a callable that takes (loss: float, parameters: List[float], model: Dataset, data: Dataset, errors: np.ndarray, residuals: np.ndarray, config: Config) and returns a float
        :type adjustment: callable
        :param description: a short description of the adjustment, used for logging
        :type description: str
        """
        self.loss_adjustments.append((adjustment, description))
        self.write_out(f"Added loss adjustment: {description}")

    def run_sampler(self,pool=None, mcmc_convergence_test=None):
        """:meta private:"""
        if self.sampler_type == 'dynesty':
            self.res, self.plot_chain, self.logprob_chain, self.log_likes, self.log_priors, self.map_params, self.mle_params = self.sample_dynesty(pool)
        if self.sampler_type == "pymultinest":
            self.res, self.plot_chain, self.logprob_chain, self.log_likes, self.log_priors, self.map_params, self.mle_params = self.sample_dynesty(pool)
        if self.sampler_type == "emcee":
            self.res, self.plot_chain, self.logprob_chain, self.log_likes, self.log_priors, self.map_params, self.mle_params = self.sample_emcee(pool, mcmc_convergence_test)
        self.max_params = self.map_params
            
    def fit(self, pool=None, mcmc_convergence_test=None):
        self.run_sampler(pool,mcmc_convergence_test)
        self.save_results()
        logging.getLogger('matplotlib').setLevel(logging.WARNING)  # suppress matplotlib debug messages
        self.visualize_results()
            
        # self.logprob_chain, self.plot_chain, self.max_params
        return self.res, self.best_fit_dict, self.best_models   
    
    def save_results(self): 
        """:meta private:"""
        # save best fit parameters
        self.best_fit_dict = dict(zip(self.param_names, self.max_params))
        # self.write_out("Best fit parameters:", self.best_fit_dict)
        try:
            param_outpath = join(self.outdir, "best_fit_params.pkl")
            save_bestfit_dict(self.best_fit_dict, param_outpath)
            self.write_out(f"Wrote best fit (MAP) parameters to {param_outpath}")
        except Exception as e:
            self.write_out(f"Writing best fit (MAP) parameters to .pkl file failed: {e}", level=logging.ERROR)
        # # save plot chain
        # plot_chain_outpath = join(self.outdir, "plot_chain.pkl")
        # with open(plot_chain_outpath, "wb") as f:
        #     pickle.dump(self.plot_chain, f)
        # self.write_out(f"Wrote plot chain to {plot_chain_outpath}")
        
        try:
            header = ','.join(self.param_names)
            plot_chain_txt_outpath = join(self.outdir, "plot_chain.txt")
            np.savetxt(fname=plot_chain_txt_outpath, X=self.plot_chain, header=header)
            self.write_out(f"Wrote plot chain to {plot_chain_txt_outpath}")
        except Exception as e:
            self.write_out(f"Writing plot chain .npy file failed: {e}", level=logging.ERROR)
            

        try:
            header = ','.join(self.param_names)
            plot_chain_txt_outpath = join(self.outdir, "raw_plot_chain.txt")
            np.savetxt(fname=plot_chain_txt_outpath, X=self._raw_chain_flat, header=header)
            self.write_out(f"Wrote raw plot chain to {plot_chain_txt_outpath}")
        except AttributeError:
            pass
        except Exception as e:
            self.write_out(f"Writing raw plot chain .npy file failed: {e}", level=logging.ERROR)
        
        try:
            logprob_chain_outpath = join(self.outdir, "log_prob_chain.txt")
            np.savetxt(logprob_chain_outpath, self.logprob_chain)
            self.write_out(f"Wrote logprob chain to {logprob_chain_outpath}")
        except Exception as e:
            self.write_out(f"Writing logprob chain .npy file failed: {e}", level=logging.ERROR)
            
        try:
            raw_logprob_chain_outpath = join(self.outdir, "raw_plot_chain.txt")
            np.savetxt(fname=raw_logprob_chain_outpath, X=self._raw_log_prob_flat)
            self.write_out(f"Wrote raw logprob chain to {raw_logprob_chain_outpath}")
        except AttributeError:
            pass
        except Exception as e:
            self.write_out(f"Writing raw logprob chain .txt file failed: {e}", level=logging.ERROR)

        try:
            log_likes_outpath = join(self.outdir, "log_like_chain.txt")
            np.savetxt(log_likes_outpath, self.log_likes)
            self.write_out(f"Wrote log likelihood chain to {log_likes_outpath}")
        except Exception as e:
            self.write_out(f"Writing log likelihood chain .txt file failed: {e}", level=logging.ERROR)
        
        try:
            log_priors_outpath = join(self.outdir, "log_prior_chain.txt")
            np.savetxt(log_priors_outpath, self.log_priors)
            self.write_out(f"Wrote log prior chain to {log_priors_outpath}")
        except Exception as e:
            self.write_out(f"Writing log prior chain .txt file failed: {e}", level=logging.ERROR)

        try:
            # make the best fit model
            model_outpath = join(self.outdir, "best_fit_models.pkl")
            self.best_models = self.make_model(self.max_params)
            with open(model_outpath, "wb") as f:
                pickle.dump(self.best_models, f)
            self.write_out(f"Wrote best fit (MAP) model to {model_outpath}")
        except Exception as e:
            self.write_out(f"Making/writing best-fit (MAP) model file failed: {e}", level=logging.ERROR)
        
        try:
            # save the post-processed data
            data_outpath = join(self.outdir, "data.pkl")
            with open(data_outpath, "wb") as f:
                pickle.dump(self.data, f)
            self.write_out(f"Wrote data to {data_outpath}")
        except Exception as e:
            self.write_out(f"Writing data .pkl file failed: {e}", level=logging.ERROR)
        
        try:
            map_p = dict(zip(self.param_names,self.map_params))
            self.write_out('Maximum a posteriori parameters:')
            for k,v in map_p.items():
                self.write_out(f'{k}: {v:.3g}')
            map_outpath = join(self.outdir,'MAP_params.json')
            with open(map_outpath,'w+') as f:
                json.dump(map_p,f)
            self.write_out(f"Wrote MAP params to {map_outpath}")
        except Exception as e:
            self.write_out(f"Writing MAP parameters .json file failed: {e}", level=logging.ERROR)
        
        try:
            mle_p = dict(zip(self.param_names,self.mle_params))
            self.write_out('Maximum likelihood parameters:')
            for k,v in mle_p.items():
                self.write_out(f'{k}: {v:.3g}')
            MLE_outpath = join(self.outdir,'MLE_params.json')
            with open(MLE_outpath,'w+') as f:
                json.dump(mle_p,f)
            self.write_out(f"Wrote MLE params to {MLE_outpath}")
        except Exception as e:
            self.write_out(f"Writing MAP parameters .json file failed: {e}", level=logging.ERROR)
            
        import corner.core
        corner_chains = corner.core._parse_input(self.plot_chain)
        self.param_w_uncert = {}
        for chain, pname in zip(corner_chains, self.param_names):
            q_lo, q_mid, q_hi = corner.quantile(chain, [0.16, 0.5, 0.84])
            q_m, q_p = q_mid - q_lo, q_hi - q_mid
            self.param_w_uncert[pname] = (q_mid, (q_p,q_m,np.std(chain)))

        self.write_out("Chain distributions:")
        for k,(med, (plus, minus, std)) in self.param_w_uncert.items():
            self.write_out(f"{k}: {med:.3g} +{plus:.3g}/-{minus:.3g}")
        
        p_with_uncert_outpath = join(self.outdir, "best_params_uncert.json")
        with open(p_with_uncert_outpath,'w+') as f:
            json.dump(self.param_w_uncert,f)
        self.write_out(f"Wrote parameter medians with uncertainties to {p_with_uncert_outpath}")
        
        # p_with_uncert_outpath = join(self.outdir, "best_params_uncert.pkl")
        # with open(p_with_uncert_outpath, "wb") as f:
        #     pickle.dump(self.param_w_uncert, f)
        # self.write_out(f"Wrote param with uncertainties to {p_with_uncert_outpath}")
    
    def visualize_results(self):
        """:meta private:"""      
        if self.sampler_type == "dynesty":
            self.visualize_dynesty()
        if self.sampler_type == "emcee":
            self.visualize_emcee()
         
        show = self.config.get("show_plots", False)
        
        try:
            plt.close()
            plot_chains_vs_priors(self.plot_chain,self.prior_transforms, self.param_names)
            savefig("posteriors_vs_priors.png", self.config, outdir=self.outdir)
            if show:
                plt.show()
        except Exception as e:
            self.write_out(f'Posteriors vs. Priors plot failed: {e}',level=logging.WARN)
        
        try:
            plt.close()
            plot_corner(self.plot_chain, self.param_names, fs=12, fs2=10)  # fs=fontsize
            savefig("corner_plot.png", self.config, outdir=self.outdir)
            if show:
                plt.show()
        except Exception as e:
            self.write_out(f'Corner / trace plot failed: {e}',level=logging.WARN)
    
    def compute_log_like(self, samples_chain, log_prob):
        log_prior = np.empty_like(log_prob)

        for i, sample in enumerate(samples_chain):
            log_p = self.log_prior(sample)
            log_prior[i] = log_p

            if not np.isfinite(log_p):
                self.write_out(f"Uh oh, sample {sample} has a log prior of {log_p}",level=logging.ERROR)

        log_like = log_prob - log_prior
        log_like[~np.isfinite(log_prior)] = -np.inf

        return log_like, log_prior
    
    def compute_log_prob(self, samples_chain, log_like):
        log_prior = np.empty_like(log_like)

        for i, sample in enumerate(samples_chain):
            log_p = self.log_prior(sample)
            log_prior[i] = log_p

            if not np.isfinite(log_p):
                self.write_out(f"Uh oh, sample {sample} has a log prior of {log_p}",level=logging.ERROR)

        log_prob = log_like + log_prior
        log_prob[~np.isfinite(log_prior)] = -np.inf

        return log_prob, log_prior
                
    
    def sample_dynesty(self, pool):
        """:meta private:"""
        target_cfg = self.config[self.config["target"]]
        dynesty_cfg = target_cfg["dynesty"]
        num_live = dynesty_cfg["nlive"]
        num_walks = dynesty_cfg["nwalks"]
        bound_method = dynesty_cfg["bound_method"]
        sample_method = dynesty_cfg["sample_method"]
        dlogz_stop_crit = dynesty_cfg["dlogz_stop_crit"]
        live_file = join(self.outdir,"dynesty_live.pkl")

        pool, managed_pool = self._resolve_process_pool(pool)
        loglike_fn = _worker_iterate if managed_pool else self.iterate
        prior_fn = _worker_prior_transform if managed_pool else self.prior_transform

        try:
            if pool is None:
                self.write_out("No pool provided.")
                size = 1
            else:
                self.write_out(f"Using pool: {pool}, size: {pool.size}")
                # dynesty breaks with a SerialPool because it thinks it has size 0.
                size = pool.size or 1

            if self.restore_from is None:
                self.write_out("Starting sampling with Dynesty")
                sampler = NestedSampler(loglike_fn, prior_fn, self.nparams, nlive=num_live,
                                        bound=bound_method, sample=sample_method, walks=num_walks, pool=pool, queue_size=size)

                sampler.run_nested(checkpoint_file=live_file, dlogz=dlogz_stop_crit, print_progress=True)
            else:
                self.write_out("Resuming sampling with Dynesty")
                sampler = NestedSampler.restore(self.restore_from, pool=pool)
                if managed_pool:
                    sampler.loglikelihood.loglikelihood = loglike_fn
                    sampler.prior_transform = prior_fn
                sampler.run_nested(resume=True, checkpoint_file=live_file, print_progress=True)

            self.write_out("Sampling complete, gathering results")

            res = sampler.results
            loglike_chain = res['logl']
            plot_chain = res.samples_equal()
            
            log_prob_chain, log_prior_chain = self.compute_log_prob(res['samples'], loglike_chain)

            map_params = res['samples'][np.argmax(log_prob_chain)]
            mle_params = res['samples'][np.argmax(loglike_chain)]
            
            return res, plot_chain, log_prob_chain, loglike_chain, log_prior_chain, map_params, mle_params
            
        finally:
            if managed_pool and pool is not None:
                pool.close()
    
    def visualize_dynesty(self):
        """:meta private:"""
        try:
            fig, axes = dyplot.traceplot(self.res, labels=self.param_names)
            savefig("traceplots.png", config=self.config, outdir=self.outdir)
        except Exception as e:
            self.write_out(f"Dynesty traceplot failed: {e}",level=logging.ERROR)
        plt.cla()
        
        try:
            fig, axes = dyplot.traceplot(self.res, labels=self.param_names, connect=True, connect_highlight=range(5))
            savefig("dyn_connected_traceplot.png", config=self.config, outdir=self.outdir)
        except Exception as e:
            self.write_out(f"Dynesty connected traceplot failed: {e}",level=logging.ERROR)
        plt.cla()
        
        try:
            fig, axes = dyplot.runplot(self.res, logplot=True)
            savefig("dyn_runplot.png", config=self.config, outdir=self.outdir)
        except Exception as e:
            self.write_out(f"Dynesty runplot failed: {e}",level=logging.ERROR)
        plt.cla()

        try:
            fig, ax = dyplot.cornerpoints(self.res, cmap='plasma',kde=False, labels=self.param_names)
            savefig("dyn_cornerpoints.png", config=self.config, outdir=self.outdir)
        except Exception as e:
            self.write_out(f"Dynesty cornerpoints failed: {e}",level=logging.ERROR)
        plt.cla()
        
        try:
            fig, ax = dyplot.cornerplot(self.res, color='blue', show_titles=True, labels=self.param_names, max_n_ticks=3, quantiles=None)
            savefig("dyn_corner.png", config=self.config, outdir=self.outdir)
        except Exception as e:
            self.write_out(f"Dynesty cornerplot failed: {e}",level=logging.ERROR)
        plt.cla()
        
    def _pymn_prior_transform(self, cube, ndim, nparams):
        unit_cube = np.array([cube[i] for i in range(ndim)])
        transformed = self.prior_transform(unit_cube)
        for i in range(ndim):
            cube[i] = transformed[i]
    
    def _pymn_iterate(self, cube, ndim, nparams):
        params = np.array([cube[i] for i in range(ndim)])
        return self.iterate(params)
    
    def sample_pymn(self):
        """:meta private:"""
        import pymultinest as pymn
        from pymultinest.analyse import Analyzer
        pid, mem = get_process_info()    
        rank, size = 0, 1 
        use_MPI = self.config["parallel"]["mpi"]
        if use_MPI:
            rank, size = get_mpi_info()
        is_main_process = rank == 0
        if is_main_process: # we are the main process
            self.write_out(f"[MAIN] PID {pid} is using {mem} GB and running with {size} processes.")
        else:  # we are a worker
            self.write_out(f"[Rank {rank}] PID {pid} is using {mem} GB")
        
        target_cfg = self.config[self.config["target"]]
        pymn_cfg = target_cfg["pymultinest"]
        
        pymn_args = ["n_live_points", "importance_nested_sampling", "multimodal", "const_efficiency_mode", "evidence_tolerance", "sampling_efficiency", "verbose"]
        kwargs = {k:pymn_cfg[k] for k in pymn_args}
        
        resume = self.restore_from is not None
        
        if resume:
            outputfiles_basename=self.restore_from
            if not outputfiles_basename.endswith("_"):
                outputfiles_basename = outputfiles_basename+"_"
            required_resume_files = [
                f"{outputfiles_basename}resume.dat",
                f"{outputfiles_basename}phys_live.points",
                f"{outputfiles_basename}post_equal_weights.dat",  # optional but often present
            ]
            if is_main_process:
                for f in [fname for fname in required_resume_files if not exists(fname)]:
                    self.write_out(f"[WARNING] can't find file '{f}' that is needed to resume a run. This may cause failure!", level=logging.WARNING)
        else:
            outputfiles_basename=join(self.outdir,self.run_prefix) 
            if not outputfiles_basename.endswith("_"):
                outputfiles_basename += "_"
        self.outputfiles_basename = outputfiles_basename
        
        # print(kwargs)
        pymn.run(LogLikelihood=self._pymn_iterate, 
                    Prior=self._pymn_prior_transform, 
                    n_dims=self.nparams,
                    use_MPI=use_MPI,
                    init_MPI=False,
                    resume=resume,
                    outputfiles_basename=outputfiles_basename,
                    **kwargs
                )
        
        if not is_main_process:
            self.write_out(f"[Rank {rank}] PID {pid} is done!")
            sys.exit(0)
        
        res = Analyzer(n_params=self.nparams, outputfiles_basename=self.outputfiles_basename)
        posterior_samples = res.get_equal_weighted_posterior()

        plot_chain = posterior_samples[:, :-1]
        loglike_chain = posterior_samples[:, -1]
        log_prob_chain, log_prior_chain = self.compute_log_prob(plot_chain, loglike_chain)

        map_params = plot_chain[np.argmax(log_prob_chain)]
        mle_params = plot_chain[np.argmax(loglike_chain)]
        
        return res, plot_chain, log_prob_chain, loglike_chain, log_prior_chain, map_params, mle_params
        
    def log_prior(self, theta):
        """ Log prior for MCMC """
        total = 0.0
        for value, transform in zip(theta, self.prior_transforms()):
            lp = transform.log_prior(value)
            if not np.isfinite(lp):
                return -np.inf
            total += lp
        return total

    def log_posterior(self, theta):
        """ Log posterior for MCMC """
        logprior = self.log_prior(theta)
        if not np.isfinite(logprior):
            return -np.inf
        loglike = self.iterate(theta)
        if not np.isfinite(loglike):
            return -np.inf
        return logprior + loglike
    
    # def set_mcmc_starting_guess(self, guess:np.ndarray):
    #     """Set the starting points for the MCMC walkers. The provided array must have shape (number of parameters, number of walkers), and the first axis must match the order of parameters. The number of walkers is set in the config. """
    #     self.mcmc_starting_guess = guess
    
    def generate_random_mcmc_guesses(self):
        """ Generate initial MCMC walker locations by drawing from the prior."""
        target_cfg = self.config[self.config["target"]]
        cfg = target_cfg["emcee"]
        nwalkers = cfg['nwalkers']
        
        points = []
        log_posts = []
        constraint_violations = []
        i = 0
        while len(points) < nwalkers:
            i+=1
            theta = []
            for p in self.params:
                theta.extend(p.prior_transform.draw(p.nvals))                    
            try:
                self.set_model_params(theta)
                for constraint in self.constraints:
                    if not constraint.validate():
                        constraint_violations.append(constraint.failure_message())
                score = self.log_posterior(theta)
                if np.isfinite(score):
                    points.append(theta)
                    log_posts.append(score)
            except ConstraintViolation:
                pass
        best = np.argmax(log_posts)                
        best_point = points[best]
        best_log_post = log_posts[best]
        self.write_out(f"While searching for MCMC initial points, generated {i} candidate points, {nwalkers} of which yielded valid posterior draws.")
        self.write_out(f"The log posteriors of the generated points had mean {np.mean(log_posts):.1f}, median {np.median(log_posts):.2f}, and standard deviation {np.std(log_posts):.2f}.")
        self.write_out(f"The best guess, with log posterior {best_log_post}, was")
        for pname, val in zip(self.param_names, best_point):
            self.write_out(f"{pname}: {val:.5f}")
        if len(constraint_violations):
            self.write_out("In the process, encountered the following constraint violations:")
            self.write_out("\n".join(constraint_violations))
        return points
    
    def generate_minimized_initial_mcmc_point(self, n_candidates=256):
        nll = lambda *args: -self.iterate(*args)
        initial, _, _ = self.generate_blind_opt_initial_mcmc_point(n_candidates)
        soln = minimize(nll, initial)
        mle = soln.x

        self.write_out("Maximum likelihood estimates:")
        for pname, val in zip(self.param_names, mle):
            self.write_out(f"{pname}: {val:.5f}")
        return mle

    def generate_blind_opt_initial_mcmc_point(self, n_candidates=256):
        """ Generate an initial MCMC guess. Randomly draws a set of candidates from a unit cube, puts them through the prior transform into physical param space, and chooses the candidate with maximum log posterior"""
        candidates_u = np.random.rand(n_candidates, self.nparams)
        candidates_x = np.array([self.prior_transform(u) for u in candidates_u])

        scores = np.full(n_candidates, -np.inf)
        constraint_violations = []
        for i, theta in enumerate(candidates_x):
            try:
                self.set_model_params(theta)
                for constraint in self.constraints:
                    if not constraint.validate():
                        constraint_violations.append(constraint.failure_message())
                scores[i] = self.log_posterior(theta)
            except ConstraintViolation:
                pass
        good = np.isfinite(scores)
        num_good = np.count_nonzero(good)
        best = np.argmax(scores)
                
        if num_good < min(int(n_candidates / 2), 32):
            raise RuntimeError(f"When looking for MCMC initial point, less than {min(int(n_candidates / 2), 32)} of the {n_candidates} initial prior draws had finite likelihoods. Please manually provide an initial guess using set_mcmc_starting_guess. During the draws, encountered the following constraint violations:\n"+"\n".join(constraint_violations))
            
        best_candidate = candidates_x[best]
        # best_score = scores[best]
        # self.write_out(f"While searching for MCMC initial point, generated {n_candidates} candidate points, {num_good} of which yielded valid posterior draws. The best guess, with log posterior {best_score}, was")
        # for pname, val in zip(self.param_names, best_candidate):
        #     self.write_out(f"{pname}: {val:.5f}")
        # if len(constraint_violations):
        #     self.write_out(f"In the process, encountered the following constraint violations:\n"+"\n".join(constraint_violations))
        return best_candidate, candidates_x, scores
                
    # def generate_optimized_mcmc_guesses(self, n_candidates=256, starting_jitter=0.01, max_attempts=10):
    #     """ Generate initial MCMC walker positions. Takes an optimized guess from :py:meth:`~sika.modeling.Sampler.generate_optimized_initial_mcmc_point` and adds a jitter, scaled by each prior, to populate the walker positions."""
    #     target_cfg = self.config[self.config["target"]]
    #     cfg = target_cfg["emcee"]
    #     nwalkers = cfg['nwalkers']
    #     best_guess, candidates, scores = self.generate_optimized_initial_mcmc_point(n_candidates)
    #     good_candidates = np.array(candidates[np.isfinite(scores)])
    #     scales = []
    #     for i, p in enumerate(self.params):
    #         if p.prior_transform.scale is not None:
    #             scale = p.prior_transform.scale
    #         else:  # estimate the scale using the spread of drawn candidates
    #             scale = np.std(good_candidates[:,i])
    #         scales.extend([scale]*p.nvals)  # param can have multiple vals if has coords
    #     scales = np.array(scales)
    #     jitter = starting_jitter
    #     for i in range(max_attempts):
    #         jitter_scale = scales * jitter
    #         self.write_out(f'jitter scale {jitter_scale.shape}: {jitter_scale}')
    #         self.write_out(f'best_guess {best_guess.shape}: {best_guess}')
    #         self.write_out(f'nwalkers: {nwalkers}')
    #         self.write_out(f'nparams: {self.nparams}')
    #         walkers = best_guess + np.random.normal(0.0, jitter_scale, size=(nwalkers, self.nparams))
    #         good = np.array([np.isfinite(self.log_posterior(w)) for w in walkers])
    #         if good.all():
    #             self.write_out("Accepted starting positions.")
    #             return walkers
    #         if np.count_nonzero(good) > 2/3 * len(good):
    #             self.write_out("Regenerating some starting positions")
    #             for i in np.where(~good):
    #                 for _ in range(max_attempts):
    #                     guess = best_guess + np.random.normal(0.0, jitter_scale, size=(self.nparams))
    #                     if np.isfinite(self.log_posterior(guess)):
    #                         walkers[i] = guess
    #                         good[i] = True
    #                         continue
    #         if good.all():
    #             self.write_out("Accepted starting positions.")
    #             return walkers
                    
    #         self.write_out(f"Rejected starting positions on iter {i}. {np.count_nonzero(~good)} out of {nwalkers} starting points were invalid. Iterating.")
    #         jitter *= 0.5
    #     raise RuntimeError(f"Unable to find viable starting positions within {max_attempts} iterations. Please supply them manually with set_mcmc_starting_guess.")
    
    def generate_minimized_mcmc_guesses(self, n_candidates=256, starting_jitter=0.01, max_attempts=10):
        target_cfg = self.config[self.config["target"]]
        cfg = target_cfg["emcee"]
        nwalkers = cfg['nwalkers']
        best_guess = self.generate_minimized_initial_mcmc_point(n_candidates)
        scales = []
        for i, p in enumerate(self.params):
            if p.prior_transform.scale is None:
                raise NotImplementedError(f'Minimized MCMC initialization only works when all prior transforms have defined scales ({p.prior_transform.dispname} does not!)')
            scale = p.prior_transform.scale
            scales.extend([scale]*p.nvals)  # param can have multiple vals if has coords
        scales = np.array(scales)
        jitter = starting_jitter
        for i in range(max_attempts):
            jitter_scale = scales * jitter
            self.write_out(f'jitter scale {jitter_scale.shape}: {jitter_scale}')
            self.write_out(f'best_guess {best_guess.shape}: {best_guess}')
            self.write_out(f'nwalkers: {nwalkers}')
            self.write_out(f'nparams: {self.nparams}')
            walkers = best_guess + np.random.normal(0.0, jitter_scale, size=(nwalkers, self.nparams))
            good = np.array([np.isfinite(self.log_posterior(w)) for w in walkers])
            if good.all():
                self.write_out("Accepted starting positions.")
                return walkers
            if np.count_nonzero(good) > 2/3 * len(good):
                self.write_out("Regenerating some starting positions")
                for i in np.where(~good):
                    for _ in range(max_attempts):
                        guess = best_guess + np.random.normal(0.0, jitter_scale, size=(self.nparams))
                        if np.isfinite(self.log_posterior(guess)):
                            walkers[i] = guess
                            good[i] = True
                            continue
            if good.all():
                self.write_out("Accepted starting positions.")
                return walkers
                    
            self.write_out(f"Rejected starting positions on iter {i}. {np.count_nonzero(~good)} out of {nwalkers} starting points were invalid. Iterating.")
            jitter *= 0.5
        raise RuntimeError(f"Unable to find viable starting positions within {max_attempts} iterations. Please supply them manually with set_mcmc_starting_guess.")
    
    def has_mcmc_converged(self, iteration, tau, previous_tau:np.ndarray):
        valid = np.isfinite(tau) & ~np.isnan(tau)
        all_valid = np.all(valid)
        
        enough_iters = tau * 100 < iteration
        all_enough_iters = np.all(enough_iters)
        
        fractional_delta_tau = np.abs(previous_tau - tau) / tau
        stable = fractional_delta_tau < 0.01
        all_stable = np.all(stable)
        
        if all_valid and all_enough_iters and all_stable:
            return True, f"MCMC has converged - done 100x more iterations than each parameter's autocorrelation time (mean tau*100={np.mean(tau)*100 :.1f}, iterations={iteration}) and each parameter's tau is stable (mean fractional delta tau = {np.mean(fractional_delta_tau) :.5f} < 0.01)."
        
        msg = "MCMC has not converged because "
        if not all_valid:
            if np.all(~all_valid):
                msg += "no parameters have valid autocorrelation values (all NaN/inf)"
            else:
                invalid_params = np.array(self.param_names)[~valid]
                param_list = ', '.join([f"'{n}' ({t:.1f})" for n,t in zip(invalid_params,tau[~valid])]) 
                msg += f"{len(invalid_params)}/{self.nparams} parameters - {param_list} - have invalid tau values"
        
        if not all_enough_iters and not np.all(~enough_iters == ~valid):  # if all of the not enough iters are because of invalid values, this msg can be skipped  (yes i know the inversion isnt necessary, but i think it makes it easier to read)
            if not all_valid:
                msg += " and "
            if np.all(~enough_iters):
                msg += f"no parameters have 100*tau < iterations (mean tau {np.mean(tau):.1f}, max tau {np.max(tau):.1f}, {iteration} iterations)"
            else:
                eff_not_enough = ~enough_iters & valid  # don't double-count invalid
                not_conv_params = np.array(self.param_names)[eff_not_enough]
                param_list = ', '.join([f"'{n}' ({t:.1f})" for n,t in zip(not_conv_params,tau[eff_not_enough])]) 
                msg += f"{len(not_conv_params)}/{self.nparams} parameters - {param_list} - do not have tau*100 > iterations ({iteration})"
        
        if not all_stable and not np.all(~all_stable == ~valid):  # if all of the not enough iters are because of invalid values, this msg can be skipped  (yes i know the inversion isnt necessary, but i think it makes it easier to read)
            if not (all_valid and all_enough_iters):
                msg += " and "
            if np.all(~stable):
                msg += f"no parameters have a stable tau (fractional change < 1%) yet: mean fractional delta tau = {np.mean(fractional_delta_tau) :.3f} and max = {np.max(fractional_delta_tau) :.3f}"
            else:
                eff_unstable = ~stable & valid  # don't double-count invalid
                not_conv_params = np.array(self.param_names)[eff_unstable]
                
                param_list = ', '.join([f"'{n}' ({fdt:.2f})" for n,fdt in zip(not_conv_params,fractional_delta_tau[eff_unstable])]) 
                msg += f"{len(not_conv_params)}/{self.nparams} parameters - {param_list} - do not have a stable tau (fractional change < 0.01) yet"
                
        return False, msg+'.'
    
    def get_manual_starting_guess(self, max_attempts=10):
        start_guess = self.flattened_guess
        if start_guess is None:
            missing_guesses = [f"'{self.short_param_names[i]}'" for i in range(len(self.params)) if self.params[i].guess is None]
            if len(missing_guesses) == len(self.params):
                raise ValueError("MCMC initialization method is 'manual' (indicating that manual starting guesses will be provided for each parameter) but no parameters have provided guesses! Provide them during initialization or manually set using each parameter's set_guess method.")
            missing_guess_str = ', '.join(missing_guesses)
            raise ValueError(f"MCMC initialization method is 'manual' (indicating that manual starting guesses will be provided for each parameter) but the following parameter(s) are missing starting guesses: {missing_guess_str}. Provide them during initialization or manually set using each parameter's set_guess method.")
        
        if not np.isfinite(self.log_posterior(start_guess)):
            raise ValueError(f"Starting guess {dict(zip(self.param_names,start_guess))} yields an invalid posterior evaluation. Please choose another guess.")
        t_config = self.config[self.config['target']]
        nwalkers = t_config['emcee']['nwalkers']
        self.write_out(f"Starting guess: {dict(zip(self.param_names,start_guess))}")
        guess = start_guess + 1e-4 * np.random.randn(nwalkers,self.nparams)
        
        for i,g in enumerate(guess):
            scale = 1e-4
            for j in range(max_attempts):
                if np.isfinite(self.log_posterior(guess[i])):
                    break
                self.write_out(f'({j+1}) Regenerating jitter {i} of manually-provided guess')
                guess[i] = start_guess + scale * np.random.randn(1,self.nparams)
                scale /= 2
            else:
                raise RuntimeError(f"Could not find a valid jittered position around the starting guess after {max_attempts} attempts.")
                
        return guess
    
    def _mcmc_param_timeseries(self, chain_full_shape, param_names):
        nparams = len(param_names)
        fig, axes = plt.subplots(nparams, figsize=(10, 7/3 * nparams), sharex=True)
        for i in range(nparams):
            ax = axes[i]
            ax.plot(chain_full_shape[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(chain_full_shape))
            ax.set_ylabel(self.param_names[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        
        return fig, axes
    
    def _make_tau_plot(self, taus, iters):
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

        ax1.plot(iters, taus, marker='o')
        ax1.plot(iters, iters / 100, linestyle='--', color='k')
        ax1.set_ylabel(r'$\tau$')
        ax1.set_title(r'MCMC Autocorrelation ($\tau$)')

        if len(taus) > 1:
            frac_change = np.abs(np.diff(taus)) / taus[1:]
            ax2.plot(iters[1:], frac_change, marker='o', linewidth=1)

        ax2.set_xlabel('iterations')
        ax2.set_ylabel(r'$|\Delta\tau|/\tau$')

        plt.tight_layout()
        return fig
    
    def sample_emcee(self, pool, convergence_test=None):
        plt.switch_backend('agg')
        if convergence_test is None:
            convergence_test = self.has_mcmc_converged
        target_cfg = self.config[self.config["target"]]
        cfg = target_cfg["emcee"]
        guess_cfg = cfg.get('initial_pos', {})
        nwalkers = cfg['nwalkers']  
        nsteps = cfg['nsteps']
        check_convergence_every = cfg.get('check_convergence_every',100)
        burn_in_factor = cfg['burn_in_factor']   
        thin_factor = cfg['thin_factor']   

        pool, managed_pool = self._resolve_process_pool(pool)
        logposterior_fn = _worker_log_posterior if managed_pool else self.log_posterior

        method = guess_cfg.get('method','minimize')
        try:
            method = EmceeGuessType(method)
        except ValueError as e:
            raise ValueError(f"Invalid emcee guess type '{method}'. Valid guess types are {[opt.value for opt in EmceeGuessType]}.") from e
        
        if method == EmceeGuessType.Manual:
            self.write_out('Using manually-provided MCMC starting points.')
            self.mcmc_starting_guess = self.get_manual_starting_guess(guess_cfg.get('max_attempts',10))
        # if method == EmceeGuessType.Optimize:
        #     self.write_out("Using optimized method to choose MCMC starting points")
        #     self.mcmc_starting_guess = self.generate_optimized_mcmc_guesses(n_candidates=guess_cfg.get('candidates',256), starting_jitter=guess_cfg.get('starting_jitter',0.01), max_attempts=guess_cfg.get('max_attempts',10))
        if method == EmceeGuessType.Random:
            self.write_out("Using random method to choose MCMC starting points")
            self.mcmc_starting_guess = self.generate_random_mcmc_guesses()
        if method == EmceeGuessType.Minimize:
            self.write_out("Using minimization method to choose MCMC starting points")
            self.mcmc_starting_guess = self.generate_minimized_mcmc_guesses(n_candidates=guess_cfg.get('candidates',256), starting_jitter=guess_cfg.get('starting_jitter',0.01), max_attempts=guess_cfg.get('max_attempts',10))
        
        try:
            if pool is None:
                self.write_out("No pool provided, running single.")
            else:
                self.write_out(f"Using pool {pool}, size: {pool.size}")

            # largely borrowed from https://emcee.readthedocs.io/en/stable/tutorials/monitor/
            
            if cfg.get("save_continuous",True):
                self.write_out('Running MCMC with HDF backend (continuous saving).')
                outfile = join(self.outdir, 'emcee_progress.h5')
                backend = emcee.backends.HDFBackend(outfile)
            else:
                self.write_out('Running MCMC with standard backend.')
                backend = emcee.backends.Backend()
                
            backend.reset(nwalkers, self.nparams)
            sampler = emcee.EnsembleSampler(nwalkers, self.nparams, logposterior_fn, pool=pool, backend=backend)
            # sampler.run_mcmc(self.mcmc_starting_guess, nsteps, progress=True)
            index = 0
            autocorr = np.empty(int(np.ceil(nsteps/check_convergence_every)))
            old_tau = np.inf
            
            progress_plot_dir = join(self.outdir, 'mcmc_progress_plots')
            makedirs(progress_plot_dir, exist_ok=True)
            self.write_out(f'Progress plots will be written to {progress_plot_dir}')
            
            for sample in sampler.sample(self.mcmc_starting_guess,iterations=nsteps, progress=True):
                if sampler.iteration % check_convergence_every:
                    continue

                # running autocorr time
                iteration = sampler.iteration
                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1

                converged, msg = convergence_test(iteration, tau, old_tau)
                self.write_out(msg)
                if converged:
                    self.write_out(f'Converged after {sampler.iteration} iterations')
                    break
                if sampler.iteration + check_convergence_every < nsteps:  # dont overwrite tau if we havent converged and we're at the end - in that case we need to keep the old val to determine convergence
                    old_tau = tau
                    
                fig, axes = self._mcmc_param_timeseries(sampler.get_chain(), self.param_names)
                plt.savefig(join(progress_plot_dir,f'param_timeseries_{index*check_convergence_every}.png'),bbox_inches="tight",dpi=300)
                
                plt.close()
                
                self._make_tau_plot(autocorr[:index], (np.arange(index)+1)*check_convergence_every)
                plt.savefig(join(progress_plot_dir,f'autocorr_{index*check_convergence_every}.png'),bbox_inches="tight",dpi=300)
                plt.close()
            
            tau = sampler.get_autocorr_time(tol=0)
            iterations = sampler.iteration
            self.write_out(f'Done sampling after {iterations} iterations (max allowed was {nsteps}).')
            converged, msg = convergence_test(iterations, sampler.get_autocorr_time(tol=0), old_tau)
            self.write_out(f"Autocorrelation time: {tau}")
            
            if not converged:
                self.write_out("DID NOT CONVERGE!",level=logging.ERROR)
                self.write_out(msg,level=logging.ERROR)
                with open(join(self.outdir,'DID_NOT_CONVERGE.txt'),'w+') as f:
                    f.write(msg)
                    
            with open(join(self.outdir,'autocorrelation_times.json'),'w+') as f:
                json.dump(dict(zip(self.param_names,tau)),f)

            self._raw_chain = sampler.get_chain()
            self._raw_chain_flat = sampler.get_chain(flat=True)
            self._raw_log_prob = sampler.get_log_prob()
            self._raw_log_prob_flat = sampler.get_log_prob(flat=True)
            if iterations > (max(tau) * burn_in_factor):
                plot_chain = sampler.get_chain(discard=int(max(tau) * burn_in_factor), thin=int(min(tau) * thin_factor), flat=True)
                log_prob_chain = sampler.get_log_prob(discard=int(max(tau) * burn_in_factor), thin=int(min(tau) * thin_factor), flat=True)
            else:
                self.write_out(f"Not chopping/thinning chains because they are too short - iterations ({iterations}) is less than the maximum tau ({max(tau)} times the burn-in factor ({burn_in_factor}, {max(tau)} * {burn_in_factor} = {max(tau) * burn_in_factor}))",level=logging.WARNING)
                plot_chain = self._raw_chain_flat
                log_prob_chain = self._raw_log_prob_flat
            
            log_like_chain, log_prior_chain = self.compute_log_like(plot_chain, log_prob_chain)

            map_params = plot_chain[np.argmax(log_prob_chain)]
            mle_params = plot_chain[np.argmax(log_like_chain)]
            
            return sampler, plot_chain, log_prob_chain, log_like_chain, log_prior_chain, map_params, mle_params
        
        finally:
            if managed_pool and pool is not None:
                pool.close()

    def _should_manage_process_pool(self) -> bool:
        parallel_cfg = self.config["parallel"]
        return (
            self.sampler_type in {"dynesty", "emcee"}
            and not parallel_cfg["mpi"]
            and parallel_cfg["processes"] > 1
        )

    def _resolve_process_pool(self, pool):
        if not self._should_manage_process_pool():
            return pool, False

        if pool is not None:
            self.write_out(
                "Recreating the process pool inside the sampler so worker processes can keep sampler state globally instead of receiving the full sampler on every task.",
                level=logging.WARNING,
            )
            pool.close()

        managed_pool = get_pool(self.config, initializer=_init_worker_sampler, initargs=(self,))
        return managed_pool, True

    def visualize_emcee(self):
        ## parameter timeseries - make a stupidly large plot
        
        fig, axes = self._mcmc_param_timeseries(self._raw_chain, self.param_names)
        savefig('mcmc_param_timeseries.png',config=self.config,outdir=self.outdir)
        