from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from sika.config import Config, logging_dir
from os.path import join, exists
import numpy as np
from dataclasses import dataclass
from sika.product import Product

@dataclass(kw_only=True)
class PTModel(Product):
    pressures: List[float]
    temperatures: List[float]

@dataclass(kw_only=True)
class PMMRModel(Product):
    pressures: List[float]
    abundances: Dict[str,List[float]]
    mmw: List[float]
    mass_fractions: Dict[str,List[float]]