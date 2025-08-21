__all__ = ["ProductIterator"]

from typing import TypeVar
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Collection

from .task import IntermediateTask
from .provider import Provider
from .product import Product

T = TypeVar('T', Product, Collection[Product], covariant=True)

class ProductIterator(IntermediateTask[Provider[T]], Provider[T], ABC):
    """ Sampler that draws from the products of the previous provider by sampling from the provided parameters."""
    
    @abstractmethod
    def sample(self, parameters:Dict[str,Any]) -> T:
        """ Draw from the products of the previous provider. """
    
    @property
    @abstractmethod
    def provided_parameters(self) -> Dict[str, List[Any]]:
        """ A dictionary of parameters and their values that tell this sampler how to sample the models. """

    def _call(self, parameters) -> T:
        return self.sample(parameters)