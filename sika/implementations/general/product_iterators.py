from typing import List, TypeVar, Any, Dict, Collection
from itertools import product

from sika.provider import Provider
from sika.product_iterator import ProductIterator
from sika.product import Product

T = TypeVar('T', bound=Product, covariant=True)


class GridIterator(ProductIterator[T]):
    """ Loops over a grid of parameters provided by the previous provider and returns a list of outputs."""
    def __init__(self, prev: Provider[T], config=None, logger=None):
        super().__init__(prev, config, logger)
        self.idx = 0
        self.grid = self._generate_grid()

    @property
    def provided_parameters(self) -> Dict[str, List[Any]]:
        return {}

    def _generate_grid(self) -> List[List[Any]]:
        if not self.prev.provided_parameters:
            return []
        vals = list(self.prev.provided_parameters.values())
        return list(product(*vals))

    def sample(self, parameters) -> T:
        # print(f"Sample {self.idx}")
        if not self.prev.provided_parameters:
            return self.prev()

        if self.idx >= len(self.grid):
            raise IndexError("No more samples available in the grid.")
        sample_params = self.grid[self.idx]
        self.idx += 1
        return self.prev(dict(zip(self.prev.provided_parameters.keys(), sample_params)))
    
    
C = TypeVar('C', Product, Collection[Product], covariant=True)


class ListIterator(ProductIterator[C]):
    """ Super simple iterator that iterates over a list of parameter dictionaries. With `loop=True`, will cycle through the list indefinitely."""
    def __init__(self, prev: Provider[C], param_list: List[Dict[str, Any]], loop=False, config=None, logger=None):
        super().__init__(prev, config, logger)
        self.idx = 0
        self.param_list = param_list
        self.loop = loop

    @property
    def provided_parameters(self) -> Dict[str, List[Any]]:
        return {}

    def sample(self, parameters) -> C:

        # TODO: the order of operations here with the index modulus seems off - may not loop correctly
        if self.idx >= len(self.param_list):
            raise IndexError("No more samples in the list.")
        sample_params = self.param_list[self.idx]
        self.idx += 1
        if self.loop:
            self.idx %= len(self.param_list)
        return self.prev(sample_params)