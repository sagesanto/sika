from os.path import join, exists
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import logging

from .profiles import PTModel, PMMRModel
from sika.provider import ProviderMiddleware

class ProfileVisualization(ProviderMiddleware[Tuple[PTModel,PMMRModel]]):
    def product_middleware(self, model: Tuple[PTModel, PMMRModel]) -> Tuple[PTModel, PMMRModel]:
        
        fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(18,6))
        fig.tight_layout(pad=3.0)
        
        pt_model, pmmr_model = model
        
        pressures, temperatures = pt_model.pressures, pt_model.temperatures
        
        # print("temps:", temperatures)
        # print("pressures:", pressures)
        ax[0].semilogy(temperatures,pressures,linewidth=3)
        ax[0].set_xlim(min(temperatures), max(temperatures))
        ax[0].set_ylabel("Pressure [bars]")
        ax[0].set_xlabel("Temperature [K]")
        ax[0].set_title(f"PT Model: {pt_model.metadata.get('model_name', 'unknown')}")
        
        pressures, abundances, mass_fractions = pmmr_model.pressures, pmmr_model.abundances, pmmr_model.mass_fractions
        
        ax[1].loglog(abundances['CH4'],pressures,label="CH4",linewidth=3)
        ax[1].loglog(abundances['CO'],pressures,label="CO",linewidth=3)
        ax[1].loglog(abundances['NH3'],pressures,label="NH3",linewidth=3)
        ax[1].loglog(abundances['CO2'],pressures,label="CO2",linewidth=3)
        ax[1].loglog(abundances['H2O'],pressures,label="H2O",linewidth=3)
        

        ax[1].legend()
        ax[1].set_ylabel("Pressure [bars]")
        ax[1].set_xlabel("VMR")
        ax[1].set_title(f"PMMR Model: {pmmr_model.metadata.get('model_name', 'unknown')}")
        

        for i in range(2):
            ax[i].minorticks_on()
            ax[i].invert_yaxis()
            ax[i].tick_params(axis='both',which='major',color="k",direction='in')
            ax[i].tick_params(axis='both',which='minor',color="k",direction='in')
        plt.show()
        
        return model


class CommonLogPressureGrid(ProviderMiddleware[Tuple[PTModel,PMMRModel]]):
    """ Makes logspace pressure grid to standardize the PT and PMMR models to the same grid."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid = None
        self.start, self.stop, self.num = None, None, None
    
    def make_common_pressure_grid(self, start:float, stop:float, num:int) -> np.ndarray:
        self.write_out(f"Creating common pressure grid from {start} to {stop} with {num} points.",level=logging.DEBUG)
        return np.logspace(start,stop,num)
    
    def get_grid(self) -> np.ndarray:
        cfg = self.config[self.config["target"]]["common_pressure_grid"]
        start, stop, num = cfg["start"], cfg["stop"], cfg["num"]
        if self.grid is None or (self.start != start or self.stop != stop or self.num != num):
            self.grid = self.make_common_pressure_grid(start, stop, num)
            self.start, self.stop, self.num = start, stop, num
        return self.grid
    
    def product_middleware(self, model: Tuple[PTModel, PMMRModel]) -> Tuple[PTModel, PMMRModel]:
        pt_model, pmmr_model = model
        grid_cfg = self.config[self.config["target"]]["common_pressure_grid"]
        grid = self.get_grid()

        pt_model.temperatures = np.interp(grid, pt_model.pressures, pt_model.temperatures)
        pt_model.pressures = grid

        pt_model.metadata["common_pressure_grid"] = grid_cfg

        for k in pmmr_model.abundances.keys():
            pmmr_model.abundances[k] = np.interp(grid, pmmr_model.pressures, pmmr_model.abundances[k])
            pmmr_model.mass_fractions[k] = np.interp(grid, pmmr_model.pressures, pmmr_model.mass_fractions[k])

        pmmr_model.mmw = np.interp(grid, pmmr_model.pressures, pmmr_model.mmw)
        pmmr_model.pressures = grid
        pmmr_model.metadata["common_pressure_grid"] = grid_cfg

        return pt_model, pmmr_model


class CommonPressureGrid(ProviderMiddleware[Tuple[PTModel,PMMRModel]]):
    """ Reads pressure grid from file to standardize the PT and PMMR models to the same grid."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid = None
        self.grid_path = None
    
    def product_middleware(self, model: Tuple[PTModel, PMMRModel]) -> Tuple[PTModel, PMMRModel]:
        pt_model, pmmr_model = model
        target_cfg = self.config[self.config["target"]]["common_pressure_grid"]
        
        grid_path = join(self.config["filestore"], target_cfg["grid_file"])
        if self.grid is None or self.grid_path != grid_path:
            self.grid = np.load(grid_path)
            self.grid_path = grid_path
        
        pt_model.temperatures = np.interp(self.grid, pt_model.pressures, pt_model.temperatures)
        pt_model.pressures = self.grid
        
        pt_model.metadata["common_pressure_grid"] = {"grid_file": self.grid_path}
        
        for k in pmmr_model.abundances.keys():
            pmmr_model.abundances[k] = np.interp(self.grid, pmmr_model.pressures, pmmr_model.abundances[k])
            pmmr_model.mass_fractions[k] = np.interp(self.grid, pmmr_model.pressures, pmmr_model.mass_fractions[k])

        pmmr_model.mmw = np.interp(self.grid, pmmr_model.pressures, pmmr_model.mmw)
        pmmr_model.pressures = self.grid
        pmmr_model.metadata["common_pressure_grid"] = {"grid_file": self.grid_path}

        return pt_model, pmmr_model