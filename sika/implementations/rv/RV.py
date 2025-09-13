from dataclasses import dataclass
from typing import List
from sika.product import DFProduct, Product
import numpy as np


@dataclass(kw_only=True)
class RV(DFProduct):
    t: np.ndarray
    rv: np.ndarray
    rv_err: np.ndarray | None = None
    
    @classmethod
    def cols(cls):
        """:meta private:"""
        return ['t', 'rv', 'rv_err']

    @classmethod
    def nullable_cols(cls) -> List[str]:
        """:meta private:"""
        return ['rv_err']
    
    def __post_init__(self):
        if self.rv_err is None:
            self.rv_err = np.zeros_like(self.rv)
    
@dataclass(kw_only=True)
class BinaryRV(Product):
    rv1: RV
    rv2: RV