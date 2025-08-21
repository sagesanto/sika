import sys
from typing import List, Collection, TypeVar, Any, Dict, Tuple, TypeVarTuple, Unpack, Generic, Union
from itertools import product
import logging
from abc import ABC, abstractmethod

from sika.provider import Provider
from sika.product import Product
from sika.task import IntermediateTask, Task

from sika.utils import NodeSpec, NodeShape, get_pool

T_co = TypeVar('T_co',bound=Product,covariant=True)

class Aggregator(Provider[Collection[T_co]], IntermediateTask[Provider[T_co]],ABC):
    """ Performs some aggregation over the outputs of a Provider, providing a collection of products as output."""
    def node_spec(self) -> NodeSpec:
        return NodeSpec(
            label=self.__class__.__name__,
            shape=NodeShape.RECT,
            color='#F4B071',
            ID = self.ID,
            edge_weight=0.5
        )

class PoolMapper(Aggregator[T_co]):
    def __init__(self, *args, progress=True, **kwargs):
        """
        A PoolMapper takes a Provider and uses a multiprocessing pool to map it over a collection of parameters
        provided to the call method. 
        It can be used to parallelize the execution of a Provider over multiple sets of parameters.
        """
        super().__init__(*args, **kwargs)
        self.do_progress = progress
    
    @property
    def provided_parameters(self) -> Dict[str, List[Any]]:
        return self.prev.provided_parameters

    def _call(self, parameters: List[Dict[str,Any]]) -> Collection[T_co]:
        with get_pool(self.config) as pool:
            self.write_out(f"PoolMapper: Using pool of size {pool.size} to map the task {self.prev.name} over {len(parameters)} parameters.", level=logging.INFO)
            pbar = None
            if self.do_progress:
                from tqdm import tqdm
                pbar = tqdm(total=len(parameters), desc=self.prev.name, unit="tasks")
                pbar.display()
            
            def callback(_):
                if pbar is not None:
                    pbar.update()
                    
            r = list(pool.map(self.prev, parameters, callback=callback))
                    
            if pbar is not None:
                pbar.close()
        return r

# intended to be used with iterators as prev task
class Exhauster(Aggregator[T_co]):
    @property
    def provided_parameters(self) -> Dict[str, List[Any]]:
        return self.prev.provided_parameters
    
    def _call(self, parameters) -> Collection[T_co]:
        r = []
        while True:
            try:
                r.append(self.prev(parameters))
            except IndexError:
                break
        return r

    
# take a list of providers and return a tuple of their outputs
V = TypeVarTuple('V')

class Splicer(Generic[Unpack[V]], Provider[Tuple[Unpack[V]]], Task):
    """A Splicer takes multiple providers and combines their outputs into a tuple. Can be called with a dictionary of parameters where keys are prefixed with the index of the provider (like so: "{index}_{key}"), or a collection of dictionaries, each corresponding to a provider."""
    def __init__(self, *prev: Unpack[Tuple[Provider[Unpack[V]]]], config=None, logger=None):
        super().__init__(config, logger)
        self.prev = prev
    
    @property
    def provided_parameters(self) -> Dict[str, List[Any]]:
        params = {}
        for i, provider in enumerate(self.prev):
            for key, value in provider.provided_parameters.items():
                params[f"{i}_{key}"] = value
        return params

    def unpack_params(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        unpacked = [{} for _ in range(len(self.prev))]
        for key, value in params.items():
            idx, k = key.split('_', 1)
            idx = int(idx)
            print(idx,k,value)
            unpacked[idx][k] = value
        print(f"Unpacked parameters: {unpacked}")
        return unpacked 

    def _call(self, parameters: Union[Dict[str,Any], Collection[Dict[str,Any]]]) -> Tuple[Unpack[V]]:
        if isinstance(parameters, dict):
            parameters = self.unpack_params(parameters)
        if len(parameters) != len(self.prev):
            raise ValueError(f"Expected {len(self.prev)} sets of parameters, got {len(parameters)}")
        return tuple(provider(param) for (provider, param) in zip(self.prev, parameters))