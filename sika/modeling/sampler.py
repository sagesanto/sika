import sys
from os import makedirs
from os.path import join, exists
import pickle
import logging
from typing import List, Union, Optional, Dict, TypeVar, Generic, Tuple, Callable
import numpy as np 
import dynesty.plotting as dyplot
from dynesty import NestedSampler
import pandas as pd

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

from sika.task import Task
from sika.modeling import Model, Dataset, DataLoader, LnLikelihood, Constraint, ConstraintViolation, AuxiliaryParameterSet, PriorTransform

# this is required or pickling of things like lambdas will not work
import dill
import dynesty.utils
dynesty.utils.pickle_module = dill
logging.getLogger('matplotlib').setLevel(logging.WARNING)  # suppress matplotlib debug messages

from sika.config import Config
from sika.utils import NodeSpec, NodeShape, save_bestfit_dict, savefig, plot_corner, get_mpi_info, get_process_info
from sika.product import Product

__all__ = ["Sampler"]

D = TypeVar('D', bound=Product, covariant=True)  # type of the data Product
M = TypeVar('M', bound=Product, covariant=True)  # type of the model Product
class Sampler(Generic[D,M], Task, ABC):
    """ Base class for samplers that fit models to data using a likelihood function. """
    
    supported_samplers = ['dynesty', "pymultinest"]

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
        """
        super().__init__(*args, **kwargs)
        if aux_params is None:
            aux_params = AuxiliaryParameterSet()
        self.aux_params = aux_params
        self.outdir = outdir
        self.run_prefix = run_prefix
        makedirs(outdir, exist_ok=True)
        self.models = models
        self.loss = loss
        self.sampler = None
        self.restore_from = restore_from
        self._data_params = data_params if data_params else {}
        self.loss_adjustments = []
        if isinstance(data, Dataset):
            self.data = data
            self.data_provider = None
        else:  # assume it's a DataLoader, will call later
            self.data_provider = data
            self.data: Dataset[D] = None
        self.constraints = constraints if constraints is not None else []

        self.params = [p for model in models for p in model.params] + self.aux_params.unfrozen
        
    @property
    def previous(self):
        if self.data_provider is not None:
            return self.models + [self.data_provider]
        return self.models
        
    @property
    def param_names(self):
        n = []
        for model in self.models:
            n.extend(model.param_names)
        n.extend(self.aux_params.all_names())
        return n
        
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
        args["aux_params"] = self.aux_params
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
        aux_params = grouped_params.pop()
        for params, model in zip(grouped_params, self.models):
            model.set_params(params)
        self.aux_params.set_values_flat(np.array(aux_params))
    
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
        for transform in self.aux_params.get_unfrozen_transforms():
            x[i] = transform(u[i])
            i += 1
        return x
    
    def prior_transforms(self) -> List[PriorTransform]:
        p = [t for m in self.models for t in m.prior_transforms()]
        p.extend(self.aux_params.get_unfrozen_transforms())
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
        self.aux_params.set_coords(data_coords)
        self.nparams_per = [model.nvals for model in self.models] + [self.aux_params.nvals]
        self.write_out("Number of parameters per model:", self.nparams_per[:-1])
        self.write_out("Number of auxiliary parameters:", self.nparams_per[-1])
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

    def add_loss_adjustment(self, adjustment: Callable[[float, List[float], Dataset, Dataset, np.ndarray, np.ndarray, Config], float], description: str):
        """Add a callable that takes the current loss, parameters, model, data, and config as input and returns a loss adjustment. Will be called at every iteration of the model and directly added to the loss

        :param adjustment: a callable that takes (loss: float, parameters: List[float], model: Dataset, data: Dataset, errors: np.ndarray, residuals: np.ndarray, config: Config) and returns a float
        :type adjustment: callable
        :param description: a short description of the adjustment, used for logging
        :type description: str
        """
        self.loss_adjustments.append((adjustment, description))
        self.write_out(f"Added loss adjustment: {description}")

    def run_sampler(self,pool):
        """:meta private:"""
        if self.sampler_type == 'dynesty':
            self.res, self.logprob_chain, self.plot_chain, self.max_params = self.sample_dynesty(pool)
        
        if self.sampler_type == "pymultinest":
            self.res, self.logprob_chain, self.plot_chain, self.max_params = self.sample_pymn()
                
    def fit(self, pool=None):
        self.run_sampler(pool)
        self.save_results()
        logging.getLogger('matplotlib').setLevel(logging.WARNING)  # suppress matplotlib debug messages
        self.visualize_results()
            
        # self.logprob_chain, self.plot_chain, self.max_params
        return self.res, self.best_fit_dict, self.best_models   
    
    def save_results(self): 
        """:meta private:"""
        # save best fit parameters
        self.best_fit_dict = dict(zip(self.param_names, self.max_params))
        self.write_out("Best fit parameters:", self.best_fit_dict)
        param_outpath = join(self.outdir, "best_fit_params.pkl")
        save_bestfit_dict(self.best_fit_dict, param_outpath)
        self.write_out(f"Wrote best fit parameters to {param_outpath}")
        
        # save plot chain
        plot_chain_outpath = join(self.outdir, "plot_chain.pkl")
        with open(plot_chain_outpath, "wb") as f:
            pickle.dump(self.plot_chain, f)
        self.write_out(f"Wrote plot chain to {plot_chain_outpath}")
        
        # save logprob chain
        logprob_chain_outpath = join(self.outdir, "logprob_chain.pkl")
        with open(logprob_chain_outpath, "wb") as f:
            pickle.dump(self.logprob_chain, f)
        self.write_out(f"Wrote logprob chain to {logprob_chain_outpath}")
        
        # make the best fit model
        model_outpath = join(self.outdir, "best_fit_models.pkl")
        self.best_models = self.make_model(self.max_params)
        with open(model_outpath, "wb") as f:
            pickle.dump(self.best_models, f)
        self.write_out(f"Wrote best fit model to {model_outpath}")
        
        # save the post-processed data
        data_outpath = join(self.outdir, "data.pkl")
        with open(data_outpath, "wb") as f:
            pickle.dump(self.data, f)
        self.write_out(f"Wrote data to {data_outpath}")
        
        import corner.core
        corner_chains = corner.core._parse_input(self.plot_chain)
        self.param_w_uncert = {}
        for chain, pname in zip(corner_chains, self.param_names):
            q_lo, q_mid, q_hi = corner.quantile(chain, [0.16, 0.5, 0.84])
            q_m, q_p = q_mid - q_lo, q_hi - q_mid
            self.param_w_uncert[pname] = (q_mid, (q_p,q_m,np.std(chain)))
        p_with_uncert_outpath = join(self.outdir, "best_params_uncert.pkl")
        with open(p_with_uncert_outpath, "wb") as f:
            pickle.dump(self.param_w_uncert, f)
        self.write_out(f"Wrote param with uncertainties to {p_with_uncert_outpath}")
        
    
    def visualize_results(self):
        """:meta private:"""      
        if self.sampler_type == "dynesty":
            self.visualize_dynesty()
         
        show = self.config.get("show_plots", False)
        
        try:
            plt.close()
            plot_corner(self.plot_chain, self.param_names, fs=12, fs2=10)  # fs=fontsize
            savefig("corner_plot.png", self.config, outdir=self.outdir)
            if show:
                plt.show()
        except Exception as e:
            print('Corner / trace plot failed')
            print(e)
        
    
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
        
        if pool is None:
            self.write_out("No pool provided.")
            size = 1
        else:
            self.write_out(f"Using pool: {pool}, size: {pool.size}")
            # this is a little silly - dynesty breaks with a SerialPool because it thinks it has size 0
            # which leads to an ambiguous error (pop from an empty list) but it comes from this
            size = pool.size or 1
        
        
        if self.restore_from is None:
            self.write_out("Starting sampling with Dynesty")
            sampler = NestedSampler(self.iterate, self.prior_transform, self.nparams, nlive=num_live,
                                    bound=bound_method, sample=sample_method, walks=num_walks, pool=pool, queue_size=size)
            
            sampler.run_nested(checkpoint_file=live_file, dlogz=dlogz_stop_crit, print_progress=True)
        else:
            self.write_out("Resuming sampling with Dynesty")
            sampler = NestedSampler.restore(self.restore_from,  pool=pool)
            sampler.run_nested(resume=True, checkpoint_file=live_file, print_progress=True)
        
        self.write_out("Sampling complete, gathering results")
        
        if pool is not None:
            pool.close() 
        
        res = sampler.results
        logprob_chain = res['logl']
        plot_chain = res.samples_equal()
        max_params = res['samples'][np.argmax(logprob_chain)]
        return res, logprob_chain, plot_chain, max_params
    
    def visualize_dynesty(self):
        """:meta private:"""
        try:
            fig, axes = dyplot.traceplot(self.res, labels=self.param_names)
            savefig("traceplots.png", config=self.config, outdir=self.outdir)
        except Exception as e:
            self.write_out(f"Dynesty visualization failed: {e}",level=logging.ERROR)
        
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
            if outputfiles_basename.endswith("_"):
                outputfiles_basename = outputfiles_basename[:-1]
            required_resume_files = [
                f"{outputfiles_basename}_resume.dat",
                f"{outputfiles_basename}_phys_live.points",
                f"{outputfiles_basename}_post_equal_weights.dat",  # optional but often present
            ]
            if is_main_process:
                for f in [fname for fname in required_resume_files if not exists(fname)]:
                    self.write_out(f"[WARNING] can't find file '{f}' that is needed to resume a run. This may cause failure!", level=logging.WARNING)
        else:
            outputfiles_basename=join(self.outdir,self.run_prefix) 
            if not outputfiles_basename.endswith("_"):
                outputfiles_basename += "_"
        self.outputfiles_basename = outputfiles_basename
        
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
        # log_evid = analyzer.get_stats()['global evidence']
        # try:
        #     logprior = np.array([prior_prt(params, non_uniform_priors, labels_all) for params in plot_chain])
        #     logprob_chain = loglike_chain + logprior
        #     print('Adding log prior worked.')
        # except Exception as e:
        #     print(f'Warning: error in logprior addition: {e}')
        #     logprob_chain = loglike_chain
        logprob_chain = loglike_chain

        max_params = plot_chain[np.argmax(logprob_chain)]
        
        return res, logprob_chain, plot_chain, max_params
        