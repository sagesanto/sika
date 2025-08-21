from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np
from petitRADTRANS.radtrans import Radtrans
import time
from sika.provider import Provider
from .spectra import Spectrum
from sika.implementations.spectroscopy.profile_models.profiles import PMMRModel, PTModel
from sika.task import IntermediateTask
import logging

from sika.utils import suppress_stdout

class pRT(Provider[Spectrum], IntermediateTask[Provider[Tuple[PTModel,PMMRModel]]]):
    """Generate a spectrum using petitRADTRANS for each provided profile model."""
    
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pressures = None
        self.radtrans = None
        self.hush = True
        
    @property
    def provided_parameters(self):
        # prt can generate a spectrum for each profile model, so provided parameters are constrained by the profile model provider
        return self.prev.provided_parameters
    
    def get_radtrans(self,pressures):
        """ Use the existing Radtrans object if it matches the pressures and configuration, otherwise create a new one. """
        target_cfg = self.config[self.config["target"]]["pRT"]
        
        r = self.radtrans
        if r is not None and self.pressures is not None:
            # print("Checking if we can reuse the existing Radtrans object...")
            pressures_match = np.array_equal(self.pressures, pressures)
            line_species_match = r.line_species == target_cfg["line_species"]
            rayleigh_species_match = r.rayleigh_species == target_cfg["rayleigh_species"]
            gas_continuum_contributors_match = r.gas_continuum_contributors == target_cfg["gas_continuum_contributors"]
            wavelength_boundaries_match = r.wavelength_boundaries == target_cfg["wavelength_boundaries"]
            line_opacity_mode_match = r.line_opacity_mode == target_cfg["line_opacity_mode"]
            if pressures_match and \
               line_species_match and \
               rayleigh_species_match and \
               gas_continuum_contributors_match and \
               wavelength_boundaries_match and \
               line_opacity_mode_match:
                    self.write_out("Reusing existing Radtrans object.", level=logging.DEBUG)
                    return r
            self.write_out("Existing Radtrans object does not match the current configuration or pressures.", level=logging.DEBUG)
            self.write_out("Pressures match:", pressures_match,level=logging.DEBUG)
            self.write_out("Line species match:", line_species_match,level=logging.DEBUG)
            self.write_out("Rayleigh species match:", rayleigh_species_match,level=logging.DEBUG)
            self.write_out("Gas continuum contributors match:", gas_continuum_contributors_match,level=logging.DEBUG)
            self.write_out("Wavelength boundaries match:", wavelength_boundaries_match,level=logging.DEBUG)
            self.write_out("Line opacity mode match:", line_opacity_mode_match,level=logging.DEBUG)
            self.write_out("Creating a new Radtrans object.", level=logging.DEBUG)
            # plt.scatter(np.arange(len(pressures)), self.pressures-pressures)
            # plt.xlabel("Index")
            # plt.ylabel("Pressure difference")
            # plt.show()
        start = time.perf_counter()
        with suppress_stdout(enabled=self.hush):
            self.radtrans = Radtrans(
                pressures=pressures,
                line_species=target_cfg["line_species"],
                rayleigh_species=target_cfg["rayleigh_species"],
                gas_continuum_contributors=target_cfg["gas_continuum_contributors"],
                wavelength_boundaries=target_cfg["wavelength_boundaries"],
                line_opacity_mode=target_cfg["line_opacity_mode"],
            )
        self.pressures = pressures
        end = time.perf_counter()
        self.write_out(f"Created new Radtrans object in {end - start:.2f} seconds.", level=logging.DEBUG)
        return self.radtrans
    
    def _setup(self):
        self.hush = not self.config["pRT"].get("noisy", False)
        with suppress_stdout(enabled=self.hush):
            from petitRADTRANS.config import petitradtrans_config_parser
            petitradtrans_config_parser.set_input_data_path(self.config["pRT"]["input_data_dir"])

    def _call(self, parameters):
        (pt_model, pmmr_model) = self.prev(parameters)
        
        pressures = np.array(pt_model.pressures)

        radtrans = self.get_radtrans(pressures)

        gravcgs = pt_model.parameters["grav"] * 100
        mass_fractions = pmmr_model.mass_fractions

        # for prt_species in rt_object.line_species:
        #     mass_fractions[prt_species] = mass_fractions.pop(prt_species.split('_')[0])

        self.write_out(f"Calculating spectra ({parameters})...", level=logging.DEBUG)
        start_time = time.perf_counter()

        wavelengths, flux, _ = radtrans.calculate_flux(
            temperatures=pt_model.temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=pmmr_model.mmw,
            reference_gravity=gravcgs,
        )
        
        end_time = time.perf_counter()
        self.write_out(f"Spectra calculated in {end_time - start_time:.2f} seconds.", level=logging.DEBUG)
        
        return Spectrum(
            parameters=pt_model.parameters,
            wlen=wavelengths*1e4,
            flux=flux,
            metadata={
                "model_name": "pRT",
                "pt_model": pt_model.metadata,
                "pmmr_model": pmmr_model.metadata,
            }
        )
