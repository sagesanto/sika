__all__ = ["Loss", "LnLikelihood"]

import numpy as np
from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod
    def __call__(self, model_output, target) -> float:
        """ Calculate the loss given model output and target data."""

class LnLikelihood(Loss):
    def __call__(self, data_errors, model_residuals) -> float:
        
        # error = calc_model.errs.flatten()
        # resid = calc_model.resids.flatten()
        error = np.concatenate(data_errors, axis=None)
        resid = np.concatenate(model_residuals, axis=None)

        all_pen = np.nansum(( np.log(2 * np.pi * error ** 2) ))
        all_chi2 = np.nansum((resid ** 2 / error ** 2))
        loglike_sum = -0.5 * ( all_chi2 + all_pen )

        # print(resid, error)
        # print(all_pen, all_chi2, loglike_sum)
        
        # print('in loglike')
        # print( np.log(2 * np.pi * error ** 2)[0:50])
        # print(resid[0:50], error[0:50])

        return loglike_sum
    
    def __str__(self):
        return "LnLikelihood"