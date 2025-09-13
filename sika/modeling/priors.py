from abc import ABC, abstractmethod
from scipy.stats import norm, truncnorm
from typing import Any, Dict, Tuple, Optional

__all__ = ["PriorTransform", "NullPriorTransform", "Uniform", "Normal"]

class PriorTransform(ABC):
    
    """
    A prior transform takes a float between [0, 1) and transforms it to the prior distribution
    of a parameter. this is where we incorporate information about the range (and possibly shape) of the prior distribution.
    See https://dynesty.readthedocs.io/en/latest/quickstart.html#prior-transforms for more.
    """

    def __call__(self, var: float) -> float:
        """
        Transform a variable from the prior space to the parameter space.
        """
        return self._call(var)

    @abstractmethod
    def _call(self, var: float) -> float:
        """
        Transform a variable from the prior space to the parameter space.
        """

    @abstractmethod
    def _to_dict(self) -> Dict[str, Any]:
        """
        Should not be called directly - call to_dict() instead.
        Convert the prior transform to a dictionary representation
        This needs to be overridden in subclasses. 
        """

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the prior transform to a dictionary representation
        """
        d = self._to_dict()
        d["class"] = self.__class__.__name__
        return d

    # @abstractmethod
    # def log_prior(self, value: float) -> float:
    #     """
    #     Return the log prior probability of a value. This is only implemented for some prior transforms.
    #     """
        


class NullPriorTransform(PriorTransform):
    """
    A :py:class:`~PriorTransform` that does not apply any transform. Primarily intended for use with frozen :py:class:`Parameters <sika.modeling.params.Parameter>`
    """

    def _call(self, var: float) -> float:
        return var

    def _to_dict(self):
        return {}
    
    # def log_prior(self, value: float) -> float:
    #     raise NotImplementedError("NullPriorTransform does not implement log_prior")


class Uniform(PriorTransform):
    """
    A :py:class:`~PriorTransform` for a uniform distribution.
    """

    def __init__(self, min_val: float, max_val: float) -> None:
        """A uniform prior between ``min_val`` and ``max_val``

        :type min_val: float
        :type max_val: float
        """
        self.min_val = min_val
        self.max_val = max_val
        super().__init__()

    def _call(self, var: float) -> float:
        return self.min_val + var * (self.max_val - self.min_val)

    def __repr__(self) -> str:
        return f"Uniform(min_val={self.min_val}, max_val={self.max_val})"

    def _to_dict(self):
        return {
            "min_val": self.min_val,
            "max_val": self.max_val,
        }
        
    # def log_prior(self, value: float) -> float:
    #     return 


class Normal(PriorTransform):
    """A :py:class:`~PriorTransform` for a normal distribution. Optionally, can be truncated to within a set of bounds."""
    
    def __init__(self, mean:float, std:float, bounds:Optional[Tuple[float,float]]=None) -> None:
        """A normal distribution centered at ``mean`` with a standard deviation of ``std``. If bounds is not None, the distribution is instead a :py:class:`scipy.stats.truncnorm` distribution.

        :param mean: the center of the distribution
        :type mean: float
        :param std: the standard deviation of the distribution
        :type std: float
        :param bounds: if provided, the prior will use a truncated normal distribution and will only draw vlaues within ``bounds``, defaults to None
        :type bounds: Optional[Tuple[float,float]], optional
        """
        self.mean = mean
        self.std = std
        self.bounds = bounds
        if bounds is not None:
            min_val, max_val = bounds
            a, b = (min_val - mean) / std, (max_val - mean) / std
            self.distr = truncnorm(a,b,loc=mean, scale=std)
        else:
            self.distr = norm(loc=mean, scale=std)

        super().__init__()

    def _to_dict(self):
        return {
            "std": self.std,
            "mean": self.mean,
            "bounds": self.bounds,
        }

    def _call(self,var:float) -> float:
        return self.distr.ppf(var)