from typing import List, Union, Dict, Any
import time
import logging
import itertools
from logging import Logger
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import xarray as xr

from sika.implementations.spectroscopy.grid_companion_model import GridCompanionModel
from sika.modeling.priors import PriorTransform
from .spectra.spectrum import Spectrum
from sika.modeling import CompositeModel, Dataset, Parameter, ParameterSet
from sika.utils import (
    parse_path,
)

__all__ = ["Gl229BParameterSet", "BinaryParameterSet", "CompositeKBinary", "SimpleBinary"]

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

class BinaryParameterSet(ParameterSet):
    def __init__(
        self,
        name: str,
        delta_teff: Union[PriorTransform, Parameter],
        delta_grav: Union[PriorTransform, Parameter],
    ):
        self.name = name
        self.delta_teff = delta_teff
        self.delta_grav = delta_grav
        self.setup()


class SimpleBinary(CompositeModel[Spectrum]):
    def __init__(
        self,
        name: str,
        binary_params: BinaryParameterSet,
        companion_a: GridCompanionModel,
        companion_b: GridCompanionModel,
        *args,
        **kwargs,
    ):
        """
        Binary model that combines two single-component models (which have had
        rv and vsini applied and have been convolved to instrument resolution).
        """
        super().__init__(name, binary_params, *args, **kwargs)
        self.companion_a = companion_a
        self.companion_b = companion_b

    @property
    def models(self):
        return [self.companion_a, self.companion_b]

    def make_model(self) -> Dataset[Spectrum]:
        """
        Combine two brown dwarf models (which have had rv and vsini applied and have been convolved to instrument resolution). 
        Assumes the models are on the same wavelength grid and simply adds their fluxes.
        """
        spec_a_set = self.companion_a.make_model()
        spec_b_set = self.companion_b.make_model()

        (joint_params, _, _) = xr.broadcast(
            self.parameter_set.as_xarray(),
            self.companion_a.parameter_set.as_xarray(),
            self.companion_b.parameter_set.as_xarray(),
        )

        all_coords = {c: np.array(v) for c, v in joint_params.coords.variables.items()}
        _coord_vals = list(all_coords.values())
        value_combinations = list(itertools.product(*_coord_vals))
        joint_dims = list(all_coords.keys())

        spectra = []
        for p in value_combinations:
            sel = dict(zip(joint_dims, p))
            spec_a = spec_a_set.values(sel)
            spec_b = spec_b_set.values(sel)

            metadata = sel  # VERY IMPORTANT
            
            metadata["individual_model_fluxes"] = {
                m.name: spec.flux
                for m, spec in zip(
                    (self.companion_a, self.companion_b), (spec_a, spec_b)
                )
            }
                        
            model_info = {}
            for comp, spec in zip((self.companion_a,self.companion_b),(spec_a,spec_b)):
                model_info[comp.name] = {
                    "classname":comp.__class__.__name__,
                    "dispname":comp.display_name,
                    "metadata": spec.metadata,
                    "flux": spec.flux,
                    "wlen": spec.wlen
                }
            metadata["models"] = model_info

            spectra.append(
                Spectrum(
                    parameters={},
                    flux=spec_a.flux + spec_b.flux,
                    wlen=spec_a.wlen,
                    metadata=metadata,
                )
            )

        ds = Dataset(spectra, dims=joint_dims)
        return ds


class CompositeKBinary(CompositeModel[Spectrum]):
    def __init__(
        self,
        name,
        binary_params: Gl229BParameterSet,
        companion_a: GridCompanionModel,
        companion_b: GridCompanionModel,
        *args,
        **kwargs,
    ):
        """
        Binary model that combines two single-component models (which have had
        rv and vsini applied and have been convolved to instrument resolution)
        according to the ratio of their k-band fluxes, a model parameter. Modeled
        spectra from each companion must have a 'k_band_flux' top-level metadata
        key.
        """
        super().__init__(name,binary_params, *args, **kwargs)
        self.companion_a = companion_a
        self.companion_b = companion_b

    @property
    def models(self):
        return [self.companion_a, self.companion_b]

    def make_model(self) -> Dataset[Spectrum]:
        """
        Combine two brown dwarf models (which have had rv and vsini applied and have been convolved to instrument resolution) according to the ratio of their k-band fluxes, a model parameter.
        """

        k_band_ratio = self.parameter_set.k_band_ratio.values()

        spec_a_set = self.companion_a.make_model()
        spec_b_set = self.companion_b.make_model()

        (joint_params, _, _) = xr.broadcast(
            self.parameter_set.as_xarray(),
            self.companion_a.parameter_set.as_xarray(),
            self.companion_b.parameter_set.as_xarray(),
        )

        all_coords = {c: np.array(v) for c, v in joint_params.coords.variables.items()}
        _coord_vals = list(all_coords.values())
        value_combinations = list(itertools.product(*_coord_vals))
        joint_dims = list(all_coords.keys())

        spectra = []
        for p in value_combinations:
            sel = dict(zip(joint_dims, p))
            spec_a = spec_a_set.values(sel)
            spec_b = spec_b_set.values(sel)
            k_band_ratio = self.parameter_set.k_band_ratio.values(sel)
            assert (
                "k_band_flux" in spec_a.metadata
            ), f"CompositeBinary requires that both components have a 'k_band_flux' top-level metadata keyword but it's missing in spectra {spec_a} from component A."
            assert (
                "k_band_flux" in spec_b.metadata
            ), f"CompositeBinary requires that both components have a 'k_band_flux' top-level metadata keyword but it's missing in spectra {spec_b} from component B."
            k_a = spec_a.metadata["k_band_flux"]
            k_b = spec_b.metadata["k_band_flux"]

            scale_factor = k_band_ratio * k_b / k_a

            metadata = sel  # VERY IMPORTANT
            
            metadata["individual_model_fluxes"] = {
                m.name: spec.flux
                for m, spec in zip(
                    (self.companion_a, self.companion_b), (spec_a, spec_b)
                )
            }
            
            metadata["k_band_flux"] = scale_factor * k_a + k_b
            metadata["scale_factor"] = scale_factor
            
            model_info = {}
            for comp, spec in zip((self.companion_a,self.companion_b),(spec_a,spec_b)):
                model_info[comp.name] = {
                    "classname":comp.__class__.__name__,
                    "dispname":comp.display_name,
                    "metadata": spec.metadata,
                    "flux": spec.flux,
                    "wlen": spec.wlen
                }
            metadata["models"] = model_info


            # we could re-normalize the flux again after applying scale factor - should we?
            spectra.append(
                Spectrum(
                    parameters={},
                    flux=scale_factor * spec_a.flux + spec_b.flux,
                    wlen=spec_a.wlen,
                    metadata=metadata,
                )
            )

        ds = Dataset(spectra, dims=joint_dims)
        return ds


