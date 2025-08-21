from os.path import join, abspath, expanduser
import numpy as np
import xarray
import petitRADTRANS.chemistry.utils as chemutils
from typing import Tuple

from sika.provider import Provider
from .profiles import PMMRModel, PTModel


class ElfOwl(Provider[Tuple[PTModel,PMMRModel]]):
    """Generate (load) a pressure-temperature profile and a volume mixing ratio profile from Elf Owl"""

    @property
    def provided_parameters(self):
        # return {
        #     "teff": np.array([700.0, 750.0, 800.0]),
        #     "grav": np.array([562.0, 1000.0, 1780.0]),
        #     "mh": np.array([0.0]),
        #     "co": np.array([0.458]),
        #     "logkzz": np.array([2.0]),
        # }
        
        return {
            "teff":  np.concatenate(
                (
                    np.arange(275, 600, 25),
                    np.arange(600, 1000, 50),
                    np.arange(1000, 2500, 100),  # runs to 2400
                )
            ),
            "grav": np.array([178.0, 316.0, 562.0, 1000.0, 1780.0, 3160.0]),
            # "grav": np.array([17.0, 31.0, 56.0, 100.0, 178.0, 316.0, 562.0, 1000.0, 1780.0, 3160.0]),
            "mh": np.array([-0.5, 0.0, 0.5]),
            # "mh": np.array([-1, -0.5, 0, 0.5, 0.7, 1]),
            "co": np.array([0.5,1.0,1.5]),
            # "co": np.array([0.5,1.0,1.5,2.5]) # in units of sun C/O
            # "co": np.array([0.22, 0.458, 0.687, 1.14]),  # note, in the paper it says 1.12 instead of 1.14, but also says that the range goes up to 1.14 so i think this is correct
            "logkzz": np.array([2.0, 4.0, 7.0, 8.0, 9.0]),
        }

    def _call(self, parameters):
        teff, grav, mh, co, logkzz = tuple(parameters.values())
        assert (
            len(parameters) == 5
        ), "ElfOwl requires 5 parameters: teff, grav, mh, co, logkzz"
        assert self.config.get("elf_owl") is not None, "ElfOwl requires 'elf_owl' section in config"
        # assert "model_dir" in self.config["elf_owl"], "ElfOwl requires key 'model_dir' in 'elf_owl' section of config"


        # co_eff = np.round(co / 0.458,1) # model files are in units of C/O=0.458
        co_eff = co

        model_dir = abspath(expanduser(self.config["elf_owl"]["model_dir"]))

        filename = f"spectra_logzz_{logkzz}_teff_{teff}_grav_{grav}_mh_{mh}_co_{co_eff}.nc"
        ds = xarray.load_dataset(join(model_dir, filename))

        pressures = ds["pressure"].values
        # print("max and min pressures:", max(pressures), min(pressures))

        pt_model = PTModel(
            parameters={
                "teff": teff,
                "grav": grav,
                "mh": mh,
                "co": co,
                "logkzz": logkzz,
            },
            pressures=np.array(pressures),
            temperatures=np.array(ds["temperature"].values),
            metadata={"model_name": "elf_owl_pt"},
        )
        cols = list(ds._variables.keys())
        species_list = [
            s for s in cols if s not in ("temperature", "flux", "pressure", "wavelength", "e-")
        ]

        vmr_dict_elfowl = {key: [] for key in species_list}
        for spe in species_list:
            vmr_dict_elfowl[spe] = ds[spe].values
        # delete electron
        # del vmr_dict_elfowl['e-']
        elfowl_mmw = chemutils.compute_mean_molar_masses_from_volume_mixing_ratios(
            vmr_dict_elfowl
        )
        mass_fractions = chemutils.volume_mixing_ratios2mass_fractions(
            vmr_dict_elfowl, mean_molar_masses=elfowl_mmw
        )

        pmr_model = PMMRModel(
            parameters={
                "teff": teff,
                "grav": grav,
                "mh": mh,
                "co": co,
                "logkzz": logkzz,
            },
            metadata={"model_name": "elf_owl_pmmr"},
            pressures=pressures,
            abundances=vmr_dict_elfowl,
            mmw=elfowl_mmw,
            mass_fractions=mass_fractions,
        )

        return (pt_model, pmr_model)
