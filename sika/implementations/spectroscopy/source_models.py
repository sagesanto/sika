from typing import List, Union, Dict, Any
import time
import logging
import itertools
from logging import Logger
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import xarray as xr

from sika.implementations.spectroscopy.utils import apply_rv_shift, convolve_to_instrument_res, rot_int_cmj
from sika.modeling.priors import PriorTransform
from sika.provider import ContinuousProvider
from .spectra import Spectrum
from sika.modeling import Model, CompositeModel, Dataset
from sika.modeling.parameter_set import EmptyParameterSet
from .companion_params import CompanionParameterSet, Gl229BParameterSet
from .crires_loader import CRIRESDataLoader
from .spectra import CRIRESSpectrum
from sika.utils import (
    groupby,
    parse_path,
)
from sika.config import Config


class GridCompanionModel(Model[Spectrum]):
    """A model for a companion star that uses a spectra provider to generate the spectrum based on parameters."""

    def __init__(
        self,
        parameter_set: CompanionParameterSet,
        spectra_provider: ContinuousProvider[Spectrum],
        *args,
        **kwargs,
    ):
        super().__init__(parameter_set, *args, **kwargs)
        self.spectra_provider = spectra_provider
        self.spectra_keys = list(self.spectra_provider.provided_parameters.keys())

    @property
    def previous(self):
        return [self.spectra_provider]

    # def configure(self, config:Union[None,Config], logger: Union[None,Logger]):
    #     super().configure(config, logger)
    #     self.spectra_provider.configure(config, logger)

    def _setup(self):
        self.model_specific_config = self.config[self.config["target"]].get(
            "grid_companion_model", {}
        )
        self.masked_ranges = self.model_specific_config.get("masked_ranges", [])
        print("Masked ranges:", self.masked_ranges)

    def get_from_provider(self, params) -> Spectrum:
        """
        Get the spectrum from the provider using the parameters in self.parameter_set.
        """
        p = {}
        for k in self.spectra_keys:
            try:
                p[k] = params[k].value
            except KeyError as e:
                raise ValueError(
                    f"Parameter {k} required by spectra provider not found in GridCompanionModel's parameter set."
                ) from e
        return self.spectra_provider(p)

    def make_model(self) -> Dataset[Spectrum]:
        """
        Generate a model spectrum for the companion using the spectra provider, then apply rv and vsini.
        lastly, convolve to instrumental resolution (config["target"]["instrument_resolution"]).
        """

        # first, slice to find the sets of param vals for the spectra provider specifically
        # unflattened, this yields a dictionary and an xr.Dataset
        spectra = []
        instrument_resolution = self.config[self.config["target"]].get(
            "instrument_resolution"
        )
        for spectra_params, remaining_params in self.parameter_set.groupby(
            self.spectra_keys
        ):
            # print("Making spectra with parameters:", spectra_params)
            spectrum = self.spectra_provider(
                spectra_params
            )  # get the spectrum from the provider
            # mask bad ranges if specified in config.
            # mask is applied before vsini and RV
            masked_flux = np.array(spectrum.flux, copy=True)
            mask = np.zeros_like(spectrum.wlen)
            for start_wlen, end_wlen in self.masked_ranges:
                mask[(spectrum.wlen >= start_wlen) & (spectrum.wlen <= end_wlen)] = 1
            mask = mask.astype(bool)
            masked_flux[np.where(mask)] = np.nan

            # now, we have a spectra for each set of parameters, so we can apply vsini and rv shifts
            # we want to group by vsini because broadening is more expensive than rv shifts
            # to do this, we use the groupby utility function because remaining_groups is an xr.Dataset, not a ParameterSet
            # this yields a dictionary that is just {'vsini': scalar}, a selector dictionary of coords, and a
            # dictionary of remaining parameter values (just rv)

            for vsini_dict, coords, rvs in groupby(
                ["vsini"], remaining_params, flatten=True
            ):
                # apply rotational broadening
                vsini = vsini_dict["vsini"]

                broadened_flux = rot_int_cmj(spectrum.wlen, masked_flux, vsini)

                for coord, rv_dict in zip(coords, rvs):
                    rv = rv_dict["rv"]
                    shifted_flux = apply_rv_shift(rv, spectrum.wlen, broadened_flux)

                    # very very important: coord makes it into the metadata we plan to attach
                    metadata = {
                        **spectrum.metadata,
                        **vsini_dict,
                        **coord,
                        **rv_dict,
                        "masked_ranges": self.masked_ranges,
                    }

                    # convolve to data resolution
                    if instrument_resolution is not None:
                        shifted_flux = convolve_to_instrument_res(
                            spectrum.wlen, shifted_flux, instrument_resolution
                        )
                        metadata["convolved_instrument_resolution"] = (
                            instrument_resolution
                        )

                    s = Spectrum(
                        parameters=spectrum.parameters,
                        wlen=spectrum.wlen,
                        flux=shifted_flux,
                        metadata=metadata,  # VERY important: attach at least the coords as metadata
                    )
                    spectra.append(s)
        # form all of the spectra up into a Dataset
        # the metadata that we attached to each Spectrum will be used to index them correctly
        ds = Dataset(spectra, dims=self.parameter_set.dims)
        return ds


class CompositeBinary(CompositeModel[Spectrum]):
    def __init__(
        self,
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
        super().__init__(binary_params, *args, **kwargs)
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


class EmpiricalCRIRESModel(Model[Spectrum]):
    """
    A model for a companion star that uses an empirical CRIRES spectrum.
    """

    def __init__(self, target_name, *args, **kwargs):
        super().__init__(EmptyParameterSet(), *args, **kwargs)
        self.empirical_spectrum: Dataset[CRIRESSpectrum] = None
        self.target_name = target_name
        self.disp_name = None
        self.ds = None
        self.loader = CRIRESDataLoader()

    @property
    def previous(self):
        return [self.loader]

    def args_to_dict(self):
        return {"target_name": self.target_name}

    def _setup(self):
        self.loader.configure(self.config, self.logger)
        self.empirical_spectrum = self.loader({"target": self.target_name})
        self.disp_name = self.config.get(self.target_name, {}).get("display_name", None)

    def make_model(self, *args, **kwargs) -> Dataset[CRIRESSpectrum]:
        """
        Generate a model spectrum for the companion using the empirical CRIRES data.
        """
        return self.empirical_spectrum

    @property
    def display_name(self):
        if self.disp_name is not None:
            return self.disp_name
        return self.target_name
