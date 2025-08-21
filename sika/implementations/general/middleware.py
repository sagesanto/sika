from sika.provider import ProviderMiddleware
from sika.product import Product

from typing import Any, Dict, Generic, TypeVar, List

T = TypeVar('T', bound=Product, covariant=True)

class ParamRestrictor(Generic[T], ProviderMiddleware[T]):
    """ Simple middeware that restricts the parameter space of previous tasks"""
    def __init__(self, allowed_parameters: Dict[str,List[Any]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allowed_parameters = allowed_parameters
        
    @property
    def provided_parameters(self):
        return self.allowed_parameters
    
    
class ParamInjector(Generic[T], ProviderMiddleware[T]):
    """ Simple middeware that injects parameters into the passed parameters before forwarding. """
    def __init__(self, inject: Dict[str,Any], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inject = inject
        
    @property
    def provided_parameters(self):
        # the parameters that we expect to recieve are the ones required by the next Provider that we are *not* injecting
        prov = self.prev.provided_parameters.copy()
        for k in self.inject.keys():
            prov.pop(k)
        return prov
         
    def parameter_middleware(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        p = parameters.copy()
        p.update(self.inject)
        return p