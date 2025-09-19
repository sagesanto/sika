__all__ = ["Loss", "LnLikelihood"]

import numpy as np
from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod
    def __call__(self, model_output, target) -> float:
        """ Calculate the loss given model output and target data."""

class LnLikelihood(Loss):
    def __call__(self, data_errors, model_residuals) -> float:
        error = np.concatenate(data_errors, axis=None)
        resid = np.concatenate(model_residuals, axis=None)

        all_chi2 = np.nansum((resid ** 2 / error ** 2))

        return -0.5 * all_chi2
    
    def __str__(self):
        return "LnLikelihood"
    

class LnLikelihoodErrPenalty(Loss):
    """A :py:class:`Loss` that sums the chi squared and a gaussian error penalty term."""
    def __call__(self, data_errors, model_residuals) -> float:
        error = np.concatenate(data_errors, axis=None)
        resid = np.concatenate(model_residuals, axis=None)

        all_pen = np.nansum(( np.log(2 * np.pi * error ** 2) ))
        all_chi2 = np.nansum((resid ** 2 / error ** 2))
        loglike_sum = -0.5 * ( all_chi2 + all_pen )
        return loglike_sum
    
    def __str__(self):
        return "LnLikelihoodErrPenalty"