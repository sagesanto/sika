import numpy as np

from typing import Union

from sika.implementations.spectroscopy.spectra.spectrum import Spectrum
from sika.implementations.spectroscopy.utils import apply_rv_shift, convolve_to_instrument_res, rot_int_cmj
from sika.modeling import Dataset, Model, Parameter, ParameterSet
from sika.modeling.priors import PriorTransform
from sika.provider import ContinuousProvider
from sika.utils import groupby

__all__ = ["CompanionParameterSet", "GridCompanionModel", "ElfOwlCompanionParameterSet","PhoenixCompanionParameterSet"]

class CompanionParameterSet(ParameterSet):
    def __init__(
        self,
        name: str,
        rv: Union[PriorTransform, Parameter],
        vsini: Union[PriorTransform, Parameter],
        **kwargs
    ):
        self.name = name
        self.rv = rv
        self.vsini = vsini
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.setup()
        
class ElfOwlCompanionParameterSet(CompanionParameterSet):
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
        super().__init__(name, rv, vsini, teff=teff, grav=grav, mh=mh, co=co, logkzz=logkzz)

class PhoenixCompanionParameterSet(CompanionParameterSet):
    def __init__(
        self,
        name: str,
        rv: Union[PriorTransform, Parameter],
        vsini: Union[PriorTransform, Parameter],
        temp: Union[PriorTransform, Parameter],
        logg: Union[PriorTransform, Parameter]
    ):
        super().__init__(name, rv, vsini, temp=temp, logg=logg)



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
                # self.write_out(vsini_dict)
                # self.write_out(coords)
                # self.write_out(rvs)
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