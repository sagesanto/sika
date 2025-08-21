import sys
from typing import List, Collection, Callable, TypeVar, Any, Dict, Tuple, TypeVarTuple, Unpack, Generic, Union
from itertools import product
import logging
from abc import ABC, abstractmethod

from sika.provider import Provider, ProviderMiddleware
from sika.product import Product


T = TypeVar('T',bound=Product,covariant=True)
S = TypeVar('S',bound=Product,covariant=True)
class MetadataMerger(ProviderMiddleware[T], ABC):
    def __init__(self, prev: Provider[T], secondary: Provider[S], config=None, logger=None):
        """ Metadata-manipulating middleware that modifies the return Product of `prev` by calling `secondary`, calling `merge` with the two `Product`s, and returning the result."""
        super().__init__(prev, config, logger)
        self.secondary = secondary
        self._last_call_params = None
    
    @property
    def previous(self):
        return [self.prev,self.secondary]
    
    def parameter_middleware(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        self._last_call_params = parameters
        return parameters
    
    @abstractmethod
    def merge(self, target:T, modifier:S):
        """Take `target`, the `Product` to be modified, and modify its metadata in some way using information from `modifier`, the secondary `Product`. Modification is performed in-place.

        :param target: the `Product` whose metadata is to be modified in-place
        :type target: T
        :param modifier: the secondary `Product` from which information for the metadata update is to be derived
        :type modifier: S
        """
    
    def product_middleware(self, model: T) -> T:
        modifier = self.secondary(self._last_call_params)
        self.merge(model, modifier)
        return model
