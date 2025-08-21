from os.path import join, exists
from typing import Dict, List, Any, Tuple

from .spectra import CRIRESSpectrum
from sika.modeling import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore",module="astropy.stats.sigma_clipping")

import numpy as np
import pandas as pd



def load_crires_spectrum(data_file: str, wavelength_file: str, wavelen_range: tuple, filter_size:int, filter_type:str, bp_sigma:int, masked_ranges:List[Tuple[int,int]]=None) -> CRIRESSpectrum:
    """Load a CRIRES spectrum from file.

    :param data_file: Path to science data
    :type data_file: str
    :param wavelength_file: Path to data from which to pull wavelengths. According to Yapeng, this should be the telluric star data, not the science data.
    :type wavelength_file: str
    :param wavelen_range: Will crop to this wavelength range
    :type wavelen_range: tuple
    :param filter_size: Size of the filter to apply
    :type filter_size: int
    :param filter_type: Type of the filter to apply - 'median' or 'gaussian'
    :type filter_type: str
    :rtype: CRIRESSpectrum
    """
    masked_ranges = masked_ranges or []
    
    df = pd.read_csv(data_file, sep=r'\s+', names=['wave', 'flux', 'err'], skiprows=1)
    if wavelength_file==data_file:
        wavedf = df
    else:
        wavedf = pd.read_csv(wavelength_file, sep=r'\s+', names=['wave', 'flux', 'err'], skiprows=1)

    wlen = np.array(wavedf['wave']) / 1000 
    flux = np.array(df['flux'])
    errors = np.array(df['err'])
    if wavelen_range is not None:
        clip_indices = np.where((wlen >= wavelen_range[0]) & (wlen <= wavelen_range[1]))
        wlen = wlen[clip_indices]
        flux = flux[clip_indices]
        errors = errors[clip_indices]
    
    return CRIRESSpectrum(
        parameters={"filter_type": filter_type, "filter_size": filter_size, "data_file": data_file, "wavelength_file": wavelength_file, "bp_sigma":bp_sigma,"masked_ranges":masked_ranges},
        wlen=wlen,
        flux=flux,
        errors=errors,
        filter_type=filter_type,
        filter_size=filter_size,
        bp_sigma=bp_sigma,
        masked_ranges=masked_ranges
    )


class CRIRESDataLoader(DataLoader[CRIRESSpectrum]):
    """
    A provider for loading CRIRES spectral data and doing some processing.
    Assumes a data directory with sub-dirs for each target, and in each a subdir for each night that is structured 
    like excalibuhr output.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @property
    def provided_parameters(self) -> Dict[str, List[Any]]:
        return {"target": []}

    def _call(self, parameters={}) -> Dataset[CRIRESSpectrum]:
        target_name = parameters.get("target", self.config["target"])
        target_cfg = self.config[target_name]
        data_cfg = target_cfg["data"]
        merged_cfg = dict(data_cfg).copy()
        merged_cfg.update(parameters)
        
        wavelen_range = merged_cfg["wavelen_range"]
        filter_size = merged_cfg["filter_size"]
        filter_type = merged_cfg["filter_type"]
        bp_sigma = merged_cfg["bp_sigma"]
        
        # confusing, but: according to Yapeng, the wavelengths should be pulled from the telluric star not the science data
        data_filename = merged_cfg["data_file"]
        wavelength_filename = merged_cfg["wavelength_file"]
        masked_ranges = merged_cfg.get("masked_ranges",[])
        basedir = merged_cfg["directory"]
        
        spectra = []
        for n in merged_cfg["nights"]:
            datadir = join(basedir,n,"out","combined")
            assert exists(datadir), f"Data directory {datadir} does not exist."
            data_file = join(datadir, data_filename)
            wavelength_file = join(datadir, wavelength_filename)
            self.write_out(f"Loading data for target {target_name} from night {n} in directory {datadir} over wavelength range {wavelen_range}")
            s = load_crires_spectrum(
                data_file, wavelength_file, wavelen_range, filter_size, filter_type, bp_sigma, masked_ranges
            )
            s.metadata["target"] = target_name
            s.metadata["display_name"] = parameters.get("display_name",target_cfg["display_name"])
            s.metadata["night"] = n
            s.metadata["directory"] = datadir
            spectra.append(s)
        
        return Dataset(spectra, dims=["night"])