__all__ = ["Model"]

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Union, Dict, TypeVar, Generic, Tuple
from logging import Logger
import numpy as np
from sika.config import Config
from .priors import PriorTransform
from .params import Parameter
from .parameter_set import ParameterSet
from sika.product import Product
from sika.task import Task
from sika.utils import NodeSpec, NodeShape
from .data import Dataset


T = TypeVar('T', bound=Product, covariant=True)
class Model(Generic[T], Task, ABC):
    """ A Task that takes a :py:class:`~modeling.parameter_set.ParameterSet` and, when called, uses the values of the `ParameterSet` to generate models of type ``T`` """
    def __init__(self, parameter_set: ParameterSet, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coords = {}
        self.parameter_set = parameter_set
        
    def node_spec(self) -> NodeSpec:
        return NodeSpec(
            label=self.__class__.__name__,
            shape=NodeShape.SQUARE,
            color="#F471F4",
            ID = self.ID,
            edge_weight=1
        ) 
        
    @abstractmethod
    def make_model(self) -> Dataset[T]:
        """
        This method should use the parameters in self.parameter_set and generate a model spectra for the source.
        This should not do additional preprocessing - that step will happen in the general model to ensure that data and model are consistent.
        """
    
    def set_coords(self, coords: Dict[str, List[Any]]) -> None:
        """
        Set the coordinates for the model. This is to set shapes of multidimensional parameters by detailing
        all possible coordinates for each dimension.
        """
        self.coords = coords
        self.parameter_set.set_coords(coords)
        
    def prior_transforms(self) -> List[PriorTransform]:
        """ Returns a list of prior transforms for the unfrozen parameters in the parameter set. """
        return self.parameter_set.get_unfrozen_transforms()
        
    @property
    def dims(self) -> List[str]:
        """
        Return the names of the dimensions of the model's parameter set, which are the keys of its coords dictionary.
        """
        return self.parameter_set.dims
    
    @property
    def ndims(self) -> int:
        """
        Return the number of dimensions of the parameter set.
        """
        return len(self.parameter_set.dims)
    
    @property
    def nvals(self) -> int:
        """
        Return the total number of values in the parameter set.
        This is the sum of all values across all parameters.
        """
        return self.parameter_set.nvals
    
    @property
    def params(self) -> List[Parameter]:
        return [p for p in self.parameter_set.unfrozen]
    
    @property
    def param_names(self) -> List[str]:
        return self.parameter_set.all_names(unfrozen_only=True)
    
    @property
    def coord_shape(self) -> Tuple[int,...]:
        """
        Return the shape of the coordinates in the parameter set.
        This is a list of integers representing the number of values in each dimension.
        """
        return self.parameter_set.coord_shape
    
    def explain_shape(self) -> str:
        s = f"{self.name} shape is determined by the parameters in the parameter set.\n"
        s += "Parameter shape:\n\t" 
        s += "\n\t".join(self.parameter_set.explain_shape().split("\n")) + "\n"
        return s
    
    def set_params(self, parameters: List[float]):
        self.parameter_set.set_values_flat(parameters)
    
    def __call__(self, parameters: List[float]) -> Dataset[T]:
        self.set_params(parameters)
        return self.make_model()
    
    @property
    def name(self) -> str:
        basename = super().name
        return f"{self.parameter_set.name} ({basename})"
    
    @property
    def display_name(self) -> str:
        return self.parameter_set.name
    
    
class CompositeModel(Model[T]):
    """A `Model` that manages and calls other `Model`s and synthesizes something from their outputs."""
    @property
    @abstractmethod
    def models(self) -> List[Model[T]]:
        """ Aggregate the constituent models that make up this composite model"""
        
    @property
    def previous(self):
        return self.models
    
    def set_coords(self, coords: Dict[str, List[Any]]) -> None:
        """
        Set the coordinates for the model. This is to set shapes of multidimensional parameters by detailing
        all possible coordinates for each dimension.
        """
        self.coords = coords
        self.parameter_set.set_coords(coords)
        for m in self.models:
            m.set_coords(coords)


    # def configure(self, config:Union[None,Config], logger: Union[None,Logger]):
    #     super().configure(config, logger)
    #     for m in self.models:
    #         m.configure(config,logger)
    
    def prior_transforms(self) -> List[PriorTransform]:
        priors = self.parameter_set.get_unfrozen_transforms()
        for m in self.models:
            priors.extend(m.prior_transforms())
        return priors
        
    @property
    def ndims(self) -> int:
        return len(self.parameter_set.dims) + sum([m.ndims for m in self.models])
    
    @property
    def nvals(self) -> int:
        return self.parameter_set.nvals + sum([m.nvals for m in self.models]) 
    
    @property
    def dims(self) -> List[str]:
        """        
        Return the unique combined dimensions of the CompositeModel's parameter set and each of its models' parameter sets, which are the keys the coords dictionary of each.
        """
        return list(set(self.parameter_set.dims + [d for m in self.models for d in m.dims]))
    
    @property
    def nparams_per(self):
        return [self.parameter_set.nvals]+[m.nvals for m in self.models]

    @property
    def params(self) -> List[Parameter]:
        return [p for p in self.parameter_set.unfrozen] + [p for m in self.models for p in m.params]

    @property
    def param_names(self) -> List[str]:
        return self.parameter_set.all_names(unfrozen_only=True) + [n for m in self.models for n in m.param_names]
    
    def explain_shape(self) -> str:
        s = f"{self.name} shape is determined by the parameters in the parameter set and the composite models.\n"
        s += "Parameter shape:\n\t" 
        s += "\n\t".join(self.parameter_set.explain_shape().split("\n")) + "\n"
        for m in self.models:
            s += f"{m.name} shape:\n\t"
            s += "\n\t".join(m.explain_shape().split("\n")) + "\n"
        return s
    
    def set_params(self,parameters:List[float]):
        self.parameter_set.set_values_flat(parameters[:self.nparams_per[0]])
        running_param_index = self.nparams_per[0]
        for i, model in enumerate(self.models):
            nparams = self.nparams_per[i+1]
            params = parameters[running_param_index:running_param_index+nparams]
            running_param_index += nparams
            model.set_params(params)