from abc import ABC, abstractmethod
import itertools
from typing import Any, Callable, List, Union, Dict, TypeVar, Generic, Tuple, Optional
import numpy as np
import xarray as xr
from sika.modeling.priors import PriorTransform
from sika.utils import groupby, joint_iter as joint_iter_generic, broadcast
from .constraint import Constraint

__all__ = ["Parameter", "RelativeParameter", "DeltaParameter", "UnitRelativeParameter", "joint_iter"]

# name, prior transform, value
class Parameter(ABC):
    """
        An n-dimensional parameter in a sampler.
            
        Simple 1D parameter: ::
        
            from sika.modeling.priors import Uniform
            param1d = Parameter("p1", Uniform(0, 10))
            param1d.set_values(1)
            param1d.get_values()
            >>> 1  # scalar value
            
        Getting values with context does not change the result, since this parameter does not vary with any coordinates: ::
            
            param1d.get_values({"a": "a_1", "b": "b_1"}) 
            >>> 1 # still
            
        Multidimensional parameters: ::
            
            param = Parameter("rv", 
                        prior_transform=Uniform(0,10),
                        varies_with=["night", "instrument", "order", "fiber"]
                    )
            
        Setting coords tells the parameter what its shape will be. coords are usually derived from the particular data being modeled at model runtime. ::
            
            param.set_coords({
                "night": ["n1", "n2", "n3"],
                "instrument": ["inst1", "inst2"],
                "order": [0, 1],
                "dither": ["f1", "f2"]  # not in varies_with, so ignored by param but fine to include for convenience
            })  
            
        .. note::
            This parameter's :py:attr:`~sika.modeling.params.Parameter.varies_with` attribute tells us that it *can* vary along the dimension "fiber", but we didn't include it in the coordinates. By excluding it, we're telling the parameter that during this modeling run the data will not vary with "fiber". As a result, the param will not have a "fiber" dimension during this run. 
        
        The resulting parameter shape reflects the coords we just set and its :py:attr:`~sika.modeling.params.Parameter.varies_with` attribute: :: 
        
            param.shape  # 3 nights, 2 instruments, 2 orders
            >>> (3,2,2)
            param.dims  # ('night', 'instrument', 'order')
            >>> ('night,instrument, order)
            param.set_values(np.random.rand(3, 2, 2))
            v = param.get_values({"night": "n2"}) # xarray.DataArray with shape (2, 2) for instrument and order at night n2
            v.values # numpy array with shape (2, 2) 
            
        Assign a 2x2 slice (instrument × order) at night = n2 (requires ordering the values correctly): ::
            
            param.set_values(np.array([[1.0, 1.1], [1.2, 1.3]]), context={"night": "n2"})
            v1 = np.array(param.get_values().values, copy=True)  # get the current values for comparison
            
        Or, do the same with an xarray DataArray (more verbose, but clearer): ::
            
            import xarray as xr
            slice_values = xr.DataArray(
                [[1.0, 1.1], [1.2, 1.3]],
                coords={"instrument": ["inst1", "inst2"], "order": [0, 1]},
                dims=["instrument", "order"]
            )
            param.set_values(slice_values, context={"night": "n2"})
            assert np.array_equal(v1, param.get_values().values)
    """
    def __init__(
        self,
        name: str,
        prior_transform: PriorTransform,
        frozen: bool = False,
        values: Optional[Union[float,xr.DataArray]] = None,
        varies_with: List[str] | None = None,
        coords: Dict[str, List[Any]] | None = None,
        dtype=float,
    ) -> None:

        self.name = name
        self.prior_transform = prior_transform
        self._init_vals = None
        self.frozen = frozen 
        self.coords = {}
        self.selectors = [{}]
        self.dtype=dtype
        if self.frozen and values is None:
            raise ValueError(
                f"Parameter {self.name} is frozen but no values are provided."
            )
            
        #: The names of the dimensions along which this parameter can vary 
        self.varies_with = varies_with if varies_with is not None else []

        if isinstance(values, xr.DataArray):
            self.dtype = values.dtype
        self._values = xr.DataArray(np.empty((1,),dtype=self.dtype))

        if coords is not None:
            if not len(self.varies_with):
                print(f"WARNING: Parameter '{self.name}' initialized with coordinates but no varies_with argument. Is this what you want?")
            self.set_coords(coords)
            
        if values is not None:
            if isinstance(values, xr.DataArray) and values.ndim > 0 and not self.coords:
                coords = dict(values.coords)
                self.set_coords(coords)
            # we do this trickery with _init_vals because setting_coords re-inserts the initial values (so that later setting a coord doesnt erase everything)
            self._init_vals = values
            self.set_values(values)
            

    def set_coords(self, coords: Dict[str, List[Any]]):
        """ Details the different values that the `Parameter`'s `varies_with` attributes can take. """

        # NOTE: it's possible that the coords dict does not contain all the keys in varies_with - this is fine.
        for k in self.varies_with:
            if k in coords:
                self.coords[k] = coords[k]
        c_keys = list(self.coords.keys())
        c_vals = list(self.coords.values())
        all_combos = list(itertools.product(*c_vals))
        self.selectors = [dict(zip(c_keys, combo)) for combo in all_combos]
        
        # set up our _values to have the correct dimensions
        if self.coords:
            self._values = xr.DataArray(np.empty(self.shape, dtype=self.dtype), coords=self.coords, dims=list(self.coords.keys()))
        else:
            self._values = xr.DataArray(np.empty((1,), dtype=self.dtype))
        # if we already had initial values, put them back in now that we have coords
        if self._init_vals is not None:
            self.set_values(self._init_vals)
        
    def __iter__(self):
        """ Iterates through tuples of (selector, corresponding value) for each of this `Parameter`'s selectors """
        for selector in self.selectors:
            yield selector, self.values(selector)
        
    @property
    def shape(self) -> tuple:
        return tuple(len(self.coords[k]) for k in self.coords) if self.coords else (1,)
    
    @property
    def nvals(self) -> int:
        """
        Return the number of individual parameter values represented by this Parameter instance.
        This is the product of the lengths of the coordinates.
        """
        return int(np.prod(self.shape))

    def set_values(self, values: Union[float, np.ndarray, xr.DataArray], context: Optional[Dict[str, Any]] = None):
        """Set the value(s) of the parameter.

        If context is provided, assigns a slice of the internal xarray using .loc.
        Otherwise, sets the entire parameter value.

        :param values: the new values of the parameter or slice
        :type values: Union[float, np.ndarray, xr.DataArray]
        :param context: the coordinates used to select the values to be set, defaults to None
        :type context: Optional[Dict[str, Any]], optional
        """
        
        # Scalar case (no coords)
        if self.shape == (1,):
            self._values = xr.DataArray(values)
            return

        # Slice assignment
        if context is not None:
            indexers = {dim: context[dim] for dim in context if dim in self.coords}
            remaining_dims = [dim for dim in self.dims if dim not in indexers]

            # Handle DataArray input for slice
            if isinstance(values, xr.DataArray):
                # Check dimensions match remaining dims
                assert set(values.dims) == set(remaining_dims), (
                    f"Slice DataArray dims {values.dims} must match remaining dims {remaining_dims}"
                )
                self._values.loc[indexers] = values
                return

            # Handle ndarray/list input for slice
            expected_shape = tuple(len(self.coords[dim]) for dim in remaining_dims)
            values_array = np.array(values).reshape(expected_shape)
            self._values.loc[indexers] = values_array
            return

        # full assignment (context is None)
        if isinstance(values, xr.DataArray):
            assert set(values.dims) == set(self.dims), (
                f"DataArray dimensions {values.dims} do not match parameter dimensions {self.dims}."
            )
            self._values = values
            return

        # full assignment with ndarray/list
        reshaped = np.array(values).reshape(self.shape)
        self._values = xr.DataArray(reshaped, coords=self.coords, dims=self.dims)
        
    def values(self, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Return the parameter value(s) corresponding to the given context
        """
        if not self.coords or context is None:
            r = self._values
        else:
            selector = {}
            for k in self.coords:
                if k in context:
                    selector[k] = context[k]
            r = self._values.sel(**selector)
        try:
            return r.item()
        except ValueError:
            return r
        
    def as_xarray(self) -> xr.DataArray:
        return self._values

    def flattened(self) -> np.ndarray:
        return self._values.values.flatten()
    
    def set_from_flat(self, flat_array: np.ndarray):
        """
        Restore parameter values from a flat array, using current shape and coords.
        """
        flat_array = np.asarray(flat_array)
        reshaped = flat_array.reshape(self.shape)
        if self.coords:
            self.set_values(xr.DataArray(reshaped, coords=self.coords, dims=list(self.coords.keys())))
        else:
            self.set_values(xr.DataArray(reshaped))

    def __repr__(self) -> str:
        return f"{self.name}(prior_transform={self.prior_transform}, coords={self.coords}, values={self.flattened()}, frozen={self.frozen})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "prior_transform": self.prior_transform.to_dict(),
            "values": self._values,
            "frozen": self.frozen,
            "varies_with": self.varies_with,
            "coords": self.coords,
            "shape": self.shape,
            "ndims": self.ndim,
            "nparams": self.nvals,
        }
    
    @property
    def dims(self) -> Tuple[str]:
        """
        Return the names of the dimensions of the parameter, which are the keys of the coords dictionary.
        """
        return tuple(self.coords.keys())

    @property
    def ndim(self) -> int:
        return len(self.shape)
    
    @property
    def all_names(self) -> List[str]:
        """
        Return a list of all names for this parameter, including the base name and any variations based on coords.
        """
        if not self.coords:
            return [self.name]
        names = []
        for selector in self.selectors:
            s_str = ", ".join(f"{k}={v}" for k, v in selector.items())
            names.append(f"{self.name} ({s_str})")
        return names

    def __lt__(self, other) -> Constraint:
        return Constraint(self, other, lambda a,b: a < b)
    
    def __gt__(self, other) -> Constraint:
        return Constraint(self, other, lambda a,b: a > b)
    

def joint_iter(*params: Parameter):
    for (x) in joint_iter_generic(*[p.as_xarray() for p in params]):
        yield x
    
    
class RelativeParameter(Parameter, ABC):
    """Base class for a parameter whose value cannot be set directly but is instead reflective of some combination the values of other :py:class:`Parameters <sika.modeling.params.Parameter>`"""
    
    def __init__(
        self,
        name: str,
        *params: Parameter
    ) -> None:
        """
        :param name: the human-readable name of the quantity that this :py:class`~sika.modeling.params.Parameter` represents
        :type name: str
        :param params: the :py:class:`Parameters <sika.modeling.params.Parameter>` that this ``RelativeParameter``'s value will be based on
        :type params: :py:class:`~sika.modeling.params.Parameter`
        """
        self.name = name
        self.params = params
        self.frozen = True

    @property
    def coords(self):
        return dict(self._values.coords)
    
    @abstractmethod
    def apply_relation(self,*params: Parameter) -> xr.DataArray:
        """ Given the two parameters from which this one is derived, perform some operation that calculates the current value of this parameter, as an xr.DataArray"""

    @property
    def selectors(self):
        coord_keys = list(self.coords.keys())
        coord_vals = list(self.coords.values())
        all_sel_values = list(itertools.product(*coord_vals))
        return [dict(zip(coord_keys, s)) for s in all_sel_values]
         
    @property
    def varies_with(self):
        return list(set([v for p in self.params for v in p.varies_with]))

    def set_coords(self, coords: Dict[str, List[Any]]):
        """:meta private:"""
        pass

    def set_values(self, values: Union[float, np.ndarray, xr.DataArray], context: Optional[Dict[str, Any]] = None):
        """:meta private:"""
        raise NotImplementedError("Can't set the value of a relative parameter.")
    
    @property
    def _values(self):
        return self.apply_relation(*self.params)
    
    def set_from_flat(self, flat_array: np.ndarray):
        """:meta private:"""
        raise NotImplementedError("Can't set the value of a relative parameter.")

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "params": self.params,
            "varies_with": self.varies_with,
            "coords": self.coords,
            "shape": self.shape,
            "ndims": self.ndim,
            "nparams": self.nvals,
        }
        

class DeltaParameter(RelativeParameter):
    """A Parameter whose value is the broadcasted sum of other :py:class:`Parameters <sika.modeling.params.Parameter>`. Useful for constraining this :py:class:`~sika.modeling.params.RelativeParameter` to be greater than or less than another :py:class:`~sika.modeling.params.Parameter` through a third :py:class:`~sika.modeling.params.Parameter` that expresses the difference between them """

    def __init__(self, name: str, *params: Parameter) -> None:
        """
        :param name: the human-readable name of the quantity that this :py:class:`~sika.modeling.params.Parameter` represents
        :type name: str
        :param params: the Parameters to sum
        """
        super().__init__(name, *params)
    
    def apply_relation(self, *params: Parameter) -> xr.DataArray:
        """:meta private:"""
        
        return sum([p.as_xarray() for p in params])
    
    def __repr__(self):
        return f"RelativeParameter("+",".join([repr(p) for p in self.params])+")"


class UnitRelativeParameter(RelativeParameter):
    """A Parameter whose value is simply the same as another :py:class:`Parameter <sika.modeling.params.Parameter>`"""

    def __init__(self, name: str, param: Parameter) -> None:
        """
        :param name: the human-readable name of the quantity that this :py:class:`~sika.modeling.params.Parameter` represents
        :type name: str
        :param param: the Parameter to mirror
        :type param: :py:class:`~sika.modeling.params.Parameter`
        """
        super().__init__(name, param)
    
    def apply_relation(self, param: Parameter) -> xr.DataArray:
        """:meta private:"""
        return param.as_xarray()
    
    def __repr__(self):
        return f"UnitRelativeParameter({repr(self.params[0])})"