from sika.modeling import Parameter, ParameterSet
from typing import Union

from sika.modeling.priors import PriorTransform

class CompanionParameterSet(ParameterSet):
    def __init__(
        self,
        name: str,
        rv: Union[PriorTransform, Parameter],
        vsini: Union[PriorTransform, Parameter],
        teff: Union[PriorTransform, Parameter],
        grav: Union[PriorTransform, Parameter],
        mh: Union[PriorTransform, Parameter],
        co: Union[PriorTransform, Parameter],
        logkzz: Union[PriorTransform, Parameter],
    ):
        self.name = name
        self.rv = rv
        self.vsini = vsini
        self.teff = teff
        self.grav = grav
        self.mh = mh
        self.co = co
        self.logkzz = logkzz
        self.setup()


class Gl229BParameterSet(ParameterSet):
    def __init__(
        self,
        name: str,
        delta_teff: Union[PriorTransform, Parameter],
        delta_grav: Union[PriorTransform, Parameter],
        k_band_ratio: Union[PriorTransform, Parameter]
    ):
        self.name = name
        self.delta_teff = delta_teff
        self.delta_grav = delta_grav
        self.k_band_ratio = k_band_ratio
        self.setup()