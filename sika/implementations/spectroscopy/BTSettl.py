import h5py
from astropy.table import Table
from sika.config import Config, config_path
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import logging

from sika.utils import parse_path
from sika.provider import Provider
from sika.implementations.spectroscopy import Spectrum

class BTSettl(Provider[Spectrum]):
    def _setup(self):
        self.model_config = self.config["BTSettl"]
        self.model_dir = parse_path(self.model_config["model_dir"])
        self.model_file = join(self.model_dir,self.model_config["model_file"])
        
        self.write_out(f"Reading BT-Settl models from {self.model_file}...",level=logging.DEBUG)
        with h5py.File(self.model_file, 'r') as hf:
            self.grid_specs = np.array(hf.get("spec"))
            self.grid_temps = np.array(hf.get("temps"))
            self.grid_loggs = np.array(hf.get("loggs"))
            self.grid_metals = np.array(hf.get("metals"))
            self.model_wvs = np.array(hf.get("wvs"))

    @property
    def provided_parameters(self):
        return {
            "teff": self.grid_temps,
            "logg": self.grid_loggs,
            "mh": self.grid_metals,
        }
        
    def _call(self, parameters):
        teff, logg, mh = parameters["teff"], parameters["logg"], parameters["mh"]
        
        if teff not in self.provided_parameters["teff"]:
            raise ValueError(f"Invalid teff: {teff}. Valid teff values are {self.grid_temps}.")
        if logg not in self.provided_parameters["logg"]:
            raise ValueError(f"Invalid logg: {logg}. Valid logg values are {self.grid_loggs}.")
        if mh not in self.provided_parameters["mh"]:
            raise ValueError(f"Invalid mh: {mh}. Valid mh values are {self.grid_metals}.")
        
        teff_idx = np.where(self.grid_temps == teff)[0]
        grav_idx = np.where(self.grid_loggs == logg)[0]
        mh_idx = np.where(self.grid_metals == mh)[0]
        
        flux = self.grid_specs[teff_idx,grav_idx,mh_idx,:].flatten()
        
        spec = Spectrum(parameters={'teff':teff,'logg':logg, 'mh':mh},
                           flux = flux,
                           wlen=self.model_wvs,
                           metadata={'model_name':'BTSettl', 'model_file':self.model_file}
                )
        
        return spec