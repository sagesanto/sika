__all__ = ["ParameterSet", "EmptyParameterSet"]

from typing import Union, Dict, List, Any, Tuple
import numpy as np
import itertools
import xarray as xr

from .priors import PriorTransform

from .params import Parameter
from sika.utils import groupby, broadcast, joint_iter as joint_iter_generic

class ParameterSet:
    unfrozen: List[Parameter] = None
    params: List[Parameter] = None
    _params: Dict[str, Parameter] = None
    coords: Dict[str, List[Any]] = None
    selectors: List[Dict[str, Any]] = None
    name: str = None

    def setup(self):
        self.unfrozen = []
        self.params = []
        self._params = {}
        for k, v in self.__dict__.items():
            if isinstance(v, PriorTransform) or isinstance(v, Parameter):
                if isinstance(v, PriorTransform):
                    p = Parameter(k, v)
                else:
                    p = v
                self.params.append(p)
                if not p.frozen:
                    self.unfrozen.append(p)
                self._params[k] = p
                setattr(self, k, p)
    
    def set_coords(self, coords: Dict[str, List[Any]]) -> None:
        """
        Set the coordinates for the parameters in the set.
        """
        coords = {c:v for c,v in coords.items() if len(v)>1}  # remove any trivial coords
        _coords = {}  # the coords passed in are all possible coords, but this will be the coords actually used by the parameters in THIS set
        for p in self.params:
            p.set_coords(coords)
            _coords.update(p.coords)  # accumulate the coords used by our parameters
        self.coords = _coords
        c_keys = list(self.coords.keys())
        c_vals = list(self.coords.values())
        all_combos = list(itertools.product(*c_vals))
        self.selectors = [dict(zip(c_keys, combo)) for combo in all_combos]
        
    def __iter__(self):
        if not self.selectors:
            yield {}, self.sel({})
            return
        for selector in self.selectors:
            yield selector, self.sel(selector)

    def sel(self, selectors: Dict[str, Any]) -> dict[str, Union[xr.DataArray, Any]]:
        """
        Select a slice of the parameter set based on the provided selectors.
        """
        return {k: self._params[k].values(selectors) for k in self._params}
    
    def as_xarray(self) -> xr.Dataset:
        return xr.Dataset({k:p.values() for k,p in self._params.items()})
    
    def groupby(self,params: List[str], flatten: bool = False):
        """ Group the dataset representation of the parameter set by the values of the specified parameters. See `utils.groupby` for details. """        
        for (x) in groupby(params, self.as_xarray(), flatten=flatten):
            yield x

    @property
    def dims(self) -> List[str]:
        """
        Return the names of the dimensions of the parameter set, which are the keys of the coords dictionary.
        """
        return list(self.coords.keys())
    
    @property
    def ndim(self) -> int:
        """
        Return the number of dimensions of the parameter set.
        """
        return len(self.dims)
    
    @property
    def nvals(self) -> int:
        """
        Return the total number of parameters in the set.
        This is the sum of the number of parameters in each unfrozen parameter.
        """
        return sum(p.nvals for p in self.unfrozen)
    
    @property
    def coord_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of the parameter set, which is the superset of the shapes of the unfrozen parameters. if all coords are 1D, this is (1,).
        """
        return tuple(len(self.coords[k]) for k in self.coords) if self.coords else (1,)
    
    def explain_shape(self) -> str:
        """
        Explain the shape of the parameter set.
        Returns a string describing the shape in terms of the coordinates.
        """
        if not self.coords:
            return f"ParameterSet has no coordinates, shape is (1,) (nvals = shape * nparams ({self.nvals}) = {self.nvals} )"
        expl = f"ParameterSet shape: {self.coord_shape}."
        for k, v in self.coords.items():
            expl += f"\n  '{k}' has {len(v)} values."
        expl += f"\nunfrozen nvals = {self.nvals}"
        for p in self.unfrozen:
            expl += f"\n  {p.name} has {p.nvals} values."
        return expl

    def set_values_direct(self, values:List[Any]) -> None:
        """
        Set the values of the parameters in the set. values must be in the correct shape and order. setting with context should be done directly
        """
        if len(values) != len(self.unfrozen):
            raise ValueError(
                f"Expected {len(self.unfrozen)} values ({self.unfrozen}), got {len(values)}"
            )
        for p, value in zip(self.unfrozen, values):
            p.set_values(value)

    def set_values_flat(self, values: np.ndarray) -> None:
        assert len(values) == self.nvals, f"Expected {self.nvals} values, got {len(values)}"
        i = 0
        for param in self.unfrozen:
            n = param.nvals
            param.set_from_flat(values[i:i+n])
            i += n
    
    def get_unfrozen_transforms(self) -> List[PriorTransform]:
        """
        Get the prior transforms of the unfrozen parameters in the set.
        """
        transforms = []
        for p in self.unfrozen:
            transforms.extend([p.prior_transform] * p.nvals)
        return transforms

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={self.params})"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the parameter set to a dictionary representation
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "params": {k: p.to_dict() for k, p in self._params.items()},
        }
        
    def all_names(self, unfrozen_only=True) -> List[str]:
        """
        Return a list of all names for the parameters in this set, including the base name and any variations based on coords.
        """
        if unfrozen_only:
            pset = self.unfrozen
        else:
            pset = self.params
        names = []
        for p in pset:
            names.extend(p.all_names)
        for i, name in enumerate(names):
            names[i] = f"{self.name}: {name}"
        return names
    
    def short_names(self, unfrozen_only=True) -> List[str]:
        """ Return a list of parameter names, not including coordinate variants. """
        if unfrozen_only:
            pset = self.unfrozen
        else:
            pset = self.params
        names = []
        for p in pset:
            names.append(p.name)
        for i, name in enumerate(names):
            names[i] = f"{self.name}: {name}"
        return names
    

def joint_iter(*param_sets: ParameterSet):
    for (x) in joint_iter_generic(*[p.as_xarray() for p in param_sets]):
        yield x


class EmptyParameterSet(ParameterSet):
    def __init__(self):
        self.name = "empty"
        self.params = {}
        self.unfrozen = []
        self.frozen = []
        self.setup()
        
    
class AuxiliaryParameterSet(ParameterSet):
    def __init__(self, name: str = "auxiliary", **params: Union[Parameter, PriorTransform]):
        self.name = name
        for k,v in params.items():
            setattr(self, k, v)
        self.setup()

