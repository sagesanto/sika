import numpy as np

from sika.implementations.rv.RV import RV, BinaryRV
from sika.implementations.rv.kepler import keplerian_function, combined_keplerian
from sika.modeling import Dataset, Model, Parameter, ParameterSet

class BinaryRVParams(ParameterSet):
    def __init__(self, name: str, amplitude_1, amplitude_2, period, eccentricity, omega_planet, tau, offset, drift):
        self.name = name
        self.amplitude_1 = amplitude_1
        self.amplitude_2 = amplitude_2
        self.period = period
        self.eccentricity = eccentricity
        self.omega_planet = omega_planet
        self.tau = tau
        self.offset = offset
        self.drift = drift
        self.setup()

class BinaryKeplerianRV(Model[BinaryRV]):
    def __init__(self, parameter_set: BinaryRVParams, t1:np.ndarray, t2:np.ndarray, *args, **kwargs):
        super().__init__(parameter_set, *args,**kwargs)
        self.t1 = t1
        self.t2 = t2
        self.t0 = min(min(t1),min(t2))

    @property
    def previous(self):
        return []
    
    def make_model(self) -> Dataset[BinaryRV]:
        rvs = []
        for sel, vals in self.parameter_set:
            amplitude_1, amplitude_2, period, eccentricity, omega_planet, tau, offset, drift = list(vals.values())
            rv1 = keplerian_function(self.t1, amplitude_1, period, eccentricity, omega_planet, tau, offset) + drift * (self.t1 - self.t0)
            rv2 = keplerian_function(self.t2, amplitude_2, period, eccentricity, omega_planet+ np.pi, tau, offset) + drift * (self.t2 - self.t0)
            
            rv_prod1 = RV(
                t=self.t1,
                rv=rv1,
                rv_err=np.zeros_like(self.t1),
                metadata={"component":"1",**sel},
                parameters=vals
            )
            rv_prod2 = RV(
                t=self.t2,
                rv=rv2,
                rv_err=np.zeros_like(self.t2),
                metadata={"component":"2",**sel},
                parameters=vals
            )
            rvs.append(BinaryRV(
                rv1=rv_prod1,
                rv2=rv_prod2,
                metadata={**sel},
                parameters=vals
            ))
            
        return Dataset(rvs, dims=self.dims)