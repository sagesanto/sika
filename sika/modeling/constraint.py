__all__ = ["BaseConstraint", "Constraint", "ListConstraint", "ConstraintViolation", "ConstraintError"]
from abc import ABC, abstractmethod
from typing import Any, List, Callable
import numpy as np

class ConstraintError(BaseException):
    """Exception indicating that something about the construction of a constraint is wrong"""
    
class ConstraintViolation(BaseException):
    """Exception indicating that a constraint was violated during sampling"""

class BaseConstraint:
    """ Base class for a constraint on an object ``val`` with respect to some object ``other`` of any type. note that the abstract :py:meth:`~BaseConstraint._compare` method here is responsible for manipulating the ``val`` and ``other`` objects before performing some actual comparison."""
    def __init__(self,val:Any,other:Any):
        self.val = val
        self.other = other
    
    @abstractmethod
    def _compare(self) -> bool:
        """Perform the comparison of :py:attr:`~BaseConstraint.val` and :py:attr:`~BaseConstraint.other` and determine whether the constraint is satisfied

        :return: a boolean that is True if the constraint is satisfied and False otherwise
        :rtype: bool
        """
    
    def failure_message(self) -> str:
        return f"Constraint {self} failed."
    
    def validate(self, raise_on_invalid=False) -> bool:
        """Determine whether or not this constraint is satisfied.

        :return: a boolean that is True if the constraint is satisfied and False otherwise
        :rtype: bool
        """
        r = self._compare()
        if not r and raise_on_invalid:
            raise ConstraintViolation(self.failure_message())
        return r

class Constraint(BaseConstraint):
    """A constraint on the value of a :py:class:`~sika.modeling.params.Parameter` with respect to a constant or another :py:class:`~sika.modeling.params.Parameter`. Each comparison uses the current 'live' value of the comparands at time of call.

        For example, the following constraint takes two Parameters, ``P1`` and ``P2``, and evaluates to True if all values of ``P1`` are less than corresponding values of ``P2`` after broadcasting.::
        
        >>> Constraint(param=P1, other=P2, comparison_func:lambda a,b: a < b)
        
        Or, if we simply wanted to constrain that the parameter P1 always have a value less than zero:::
        
        >>> Constraint(param=P1, other=0, comparison_func:lambda a,b: a < b)
        
    """
    def __init__(self, param, other:Any,comparison_func, message_formatter:Callable[[Any,Any],str]=lambda a,b: f"{a} ? {b}"):
        """A constraint on the value of a :py:class:`~sika.modeling.params.Parameter` with respect to a constant or another :py:class:`~sika.modeling.params.Parameter`.

        :param param: the Parameter whose value should be constrained
        :type param: :py:class:`~sika.modeling.params.Parameter`
        :param other: the other object against which the Parameter will be constrained. Can be a :py:class:`~sika.modeling.params.Parameter`.
        :type other: Any
        :param comparison_func: A function that has one of two signatures: if ``other`` is not a Parameter, expects one value of ``param`` and the object ``other`` and returns a boolean determining whether or not the constraint is met. If both ``param`` and ``other`` are of type :py:class:`~sika.modeling.params.Parameter`, this method should expect to compare one value (not the whole :py:class:`~sika.modeling.params.Parameter`) from each of the two parameters. This comparison will be pairwise performed on all broadcasted parameter value pairs, but this function should only deal with the underlying values stored by the Parameters.  
        :type comparison_func: _type_
        :raises ConstraintError: _description_
        """
        super().__init__(val=param,other=other)
        self.comparison_func = comparison_func

        # this constraint is 'relative' if other is a parameter
        self.relative = hasattr(other, "varies_with")
        try:                            
            extra = set(other.varies_with) - set(param.varies_with) 
            if len(extra):
                raise ConstraintError(f"Constraint is not valid: comparison parameter '{other.name}' varies along {extra} but the constrained parameter '{param.name}' does not.")
            self.relative = True    
        except AttributeError:  # this means the other is not a parameter
            pass
        
        self.message_formatter = message_formatter
        
    def failure_message(self) -> str:
        return f"Constraint failed: {self.message_formatter(self.val, self.other)}"
    
    def _compare(self):
        # if self.relative:
        # for selector, (value, other) in joint_iter(self.param.)
        
        for selector, value in self.val:
            if self.relative:  # other val is a parameter
                try:
                    other = self.other.values(selector)
                except Exception as e:
                    raise ConstraintError(f"Constraint is not valid (parameter coords): error when broadcasting parameters {self.val} and {self.other}: {e}") from e
            else:
                other = self.other
            # print(f"comparing {value} and {other}")
            if not self.comparison_func(value, other):
                return False
        # print("done comparing")
        return True
    
class ListConstraint(BaseConstraint):
    """A constraint on the value of a `Parameter` with respect to a list of constants. The `validate` method assesses whether the constraint is satisfied the current value(s) of the constrained `Parameter`."""
    def __init__(self,param,other:List[Any],comparison_func:Callable[[Any,Any],bool]):
        """A constraint on the value of a `Parameter` with respect to a list of constants.

        :param param: the `Parameter` to be constrained. if its values are 1-dimensional, they will be compared to each value in `other`. otherwise, it must flatten to the same shape as `other`.
        :type param: `Parameter`
        :param other: a list of values to compare with `param`. this constraint is only satisfied if all comparisons are true.
        :type other: List[Any]
        :param comparison_func: the function to use for comparing *values* of `param` with values of `other`.
        :type comparison_func: Callable[[Any,Any],bool]
        """
        super().__init__(val=param,other=other)
        self.comparison_func = comparison_func
    
    def _compare(self):
        vals = self.val.flattened()
        # if parameter is 1D, we'll compare that 1 value with each of the values in other 
        if len(vals) == 1:  
            vals = [vals[0] for _ in self.other]
            
        if len(vals) != len(self.other):
            raise ConstraintError(f"Constraint invalid: length of flattened Parameter values ({len(vals)}) does not match length of comparison values ({len(self.other)})")
        
        return bool(np.all([self.comparison_func(v,o) for v,o in zip(vals, self.other)]))
