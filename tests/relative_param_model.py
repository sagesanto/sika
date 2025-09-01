import os

from sika.modeling.priors import Normal, PriorTransform, Uniform
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
# from mpi4py import rc
# rc.initialize = False

# from mpi4py import MPI
# # MPI.Init()

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

# print(f"[Rank {rank}] size={size}")

import faulthandler
faulthandler.enable()

import sys
from os import makedirs
from os.path import join
from typing import List, Callable, Tuple, Union, Optional, Dict
import numpy as np
import dynesty.plotting as dyplot
import itertools
from dynesty import NestedSampler
import pickle
import logging

# from schwimmbad import MPIPool
from dataclasses import dataclass
import matplotlib.pyplot as plt
from logging import Logger

from sika.implementations.spectroscopy.spectra.spectrum import Spectrum
from sika.provider import Provider, ContinuousProvider, Product
from sika.task import Task
from sika.modeling import (
    Sampler,
    Model,
    Dataset,
    DataLoader,
    Parameter,
    ParameterSet,
    LnLikelihood,
    CompositeModel,
    DeltaParameter
)

from sika.config import Config, configure_logger, config_path
from sika.utils import save_bestfit_dict, savefig, plot_corner, get_mpi_info, get_process_info, get_sampler_pool

def nearly_equal(a,b,epsilon=1e-8):
    return abs(a-b) < epsilon


@dataclass(kw_only=True)
class MockProduct(Product):
    val: int
    
    @classmethod
    def param_names(cls) -> List[str]:
        return ['A', 'B']
    
class MockParameterSet(ParameterSet):
    def __init__(self, name: str, A: Union[PriorTransform, Parameter], B: Union[PriorTransform, Parameter]):
        self.name = name
        self.A = A
        self.B = B
        self.setup()
    
class DeltaParameterSet(ParameterSet):
    def __init__(self, name: str, Delta: Union[PriorTransform, Parameter]):
        self.name = name
        self.Delta = Delta
        self.setup()
    
class MockModel(Model[MockProduct]):
    def __init__(self, parameter_set: MockParameterSet, *args, **kwargs):
        super().__init__(parameter_set, *args, **kwargs)
        
    @property
    def previous(self):
        return []

    def make_model(self) -> Dataset[MockProduct]:
        products = []
        for selector, pset in self.parameter_set:
            prod = MockProduct(parameters=pset, val=sum(pset.values()), metadata=selector)
            products.append(prod)

        return Dataset(products, dims=self.parameter_set.dims)
    
class DeltaMockModel(CompositeModel[MockProduct]):
    def __init__(self, parameter_set: DeltaParameterSet, model:Model[MockProduct], *args, **kwargs):
        super().__init__(parameter_set, *args, **kwargs)
        self._model = model
    
    @property
    def models(self):
        return [self._model]
    
    def make_model(self) -> Dataset[MockProduct]:
        products_ds = self.models[0].make_model()
        pid, mem = get_process_info()    
        rank, size = 0, 1 
        use_MPI = self.config["parallel"]["mpi"]
        if use_MPI:
            rank, size = get_mpi_info()
        
        for sel, prod in products_ds:
            delta = self.parameter_set.Delta.values(sel)
            mock_params = self.models[0].parameter_set.sel(sel)
            expected_val = 2*self.models[0].parameter_set.A.values(sel) + delta
            if nearly_equal(prod.val, expected_val):
                pass
                # self.write_out(f"[Rank {rank}, PID {pid}] got expected value ({mock_params}, delta {delta} yields {prod.val})")
            else:
                pass
                # self.write_out(f"[Rank {rank}, PID {pid}] DID NOT get expected value {expected_val}. Instead, params {mock_params} and delta {delta} yielded {prod.val}", level=logging.ERROR)
        
        return products_ds

class DeltaSampler(Sampler[MockProduct,MockProduct]):
    def _make_model(self):
        return self.models[0].make_model()
    
    def get_errors_and_residuals(self, modeled_ds: Dataset[MockProduct]) -> Tuple[np.ndarray, np.ndarray]:
        errors = []
        residuals = []
        for selector, data_prod in self.data:
            # select the modeled product for this selector
            model_prod = modeled_ds.values(selector)

            # calculate the residual and loss
            resid = data_prod.val - model_prod.val
            err = np.random.normal(0.1,0.1)
            
            errors.append(err)
            residuals.append(resid)
        
        errors = np.array(errors)
        residuals = np.array(residuals)
        return errors, residuals
        
    
    def make_model(self,parameters:List[float]) -> Dataset[MockProduct]:
        model_ds = super().make_model(parameters)
        products_ds = self.models[0].make_model()
        pid, mem = get_process_info()    
        rank, size = 0, 1 
        use_MPI = self.config["parallel"]["mpi"]
        if use_MPI:
            rank, size = get_mpi_info()
            
        # live_a_id = id(self.models[0].parameter_set.A)
        # live_b_id = id(self.models[0].parameter_set.B)
        # live_d_id = id(self.parameter_set.Delta)
        
        # dead_a_id = id(self.models[0].parameter_set.B.params[0])
        # dead_a_id = id(self.models[0].parameter_set.B.params[1])
        
        for sel, prod in model_ds:
            delta = parameters[0]
            mock_params = parameters[1:]
            expected_val = 2*mock_params[0] + delta
            if nearly_equal(prod.val, expected_val):
                pass
                # self.write_out(f"[Rank {rank}, PID {pid}] got expected value") # ({mock_params}, delta {delta} yields {prod.val})")
            else:
                pass
                # self.write_out(f"[Rank {rank}, PID {pid}] DID NOT get expected value {expected_val}. Instead, params {mock_params} and delta {delta} yielded {prod.val}", level=logging.ERROR)
        
        return model_ds


if __name__ == "__main__":
    config = Config(config_path)
    config["target"] = "sine_model"

    logger = configure_logger("test_delta")

    data = Dataset([MockProduct(parameters={},metadata={},val=5)])
            
    delta_params = DeltaParameterSet(
        name="delta param set",
        Delta=Uniform(0,5)
    )
    
    A = Parameter("A", Uniform(0,5))
    # B = Parameter("B", Uniform(0,5))
    mock_params = MockParameterSet(
        "mock param set",
        A=A,
        # B=B
        B=DeltaParameter("B",A,delta_params.Delta)
    )

    mock_model = MockModel(mock_params)

    model = DeltaMockModel(delta_params,mock_model)
    
    pool = get_sampler_pool(config)
    
    
    sampler = DeltaSampler("test_relative_params","./test_out",data,[model])

    sampler.configure(config, logger)
    sampler.fit(pool)