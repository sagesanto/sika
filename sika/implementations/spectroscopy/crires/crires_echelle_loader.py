from os.path import join, exists
from typing import Dict, List, Any, Tuple, Optional

from sika.implementations.spectroscopy.crires.crires_echelle_spectrum import CRIRESEchelleSpectrum, CRIRESOrder
from sika.implementations.spectroscopy.crires import CRIRESSpectrum
from sika.modeling import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore",module="astropy.stats.sigma_clipping")

import numpy as np
import pandas as pd

__all__ = ["load_crires_echelle_spectrum", "CRIRESEchelleDataLoader"]

def load_crires_echelle_spectrum(data_file: str, wavelength_file: str, wavelen_range: Tuple[float,float] | None, sigma_clip, remove_continuum, filter_size:int, filter_type:str, bp_sigma:float, masked_ranges:Optional[List[Tuple[float,float]]]=None, metadata=None) -> CRIRESEchelleSpectrum:
    """Load a CRIRES spectrum from file. Loaded spectra will be continuum subtracted and normalized.

    :param data_file: Path to science data
    :type data_file: str
    :param wavelength_file: Path to data from which to pull wavelengths. According to Yapeng, this should be the telluric star data, not the science data.
    :type wavelength_file: str
    :param wavelen_range: If None, no crop performed. If tuple (start, stop), will crop to only include this wavelength range.
    :type wavelen_range: tuple
    :param sigma_clip: whether to clean the continuum with a sigma clip of sigma=bp_sigma
    :type sigma_clip: bool
    :param remove_continuum: whether to remove the continuum using the specified filter
    :type remove_continuum: bool
    :param filter_size: Size of the filter to apply
    :type filter_size: int
    :param filter_type: Type of the filter to apply - 'median' or 'gaussian'
    :type filter_type: str
    :param bp_sigma: sigma threshold to use for outlier flux removal
    :type bp_sigma: float
    :param masked_ranges: collection of (start, stop) tuples of masked-out wavelength ranges to remove
    :type masked_ranges: Optional[List[Tuple[float,float]]]
    :param metadata: dictionary of metadata to attach to each loaded :py:class:`~CRIRESOrder`
    :type metadata: dict[str,Any]
    :rtype: CRIRESEchelleSpectrum
    """
    masked_ranges = masked_ranges or []
    metadata = metadata or {}
    
    df = pd.read_csv(data_file, sep=r'\s+', names=['wave', 'flux', 'err'], skiprows=1)
    if wavelength_file==data_file:
        wavedf = df
    else:
        wavedf = pd.read_csv(wavelength_file, sep=r'\s+', names=['wave', 'flux', 'err'], skiprows=1)

    wlen = np.array(wavedf['wave']) / 1000 
    flux = np.array(df['flux'])
    errors = np.array(df['err'])
    if wavelen_range is not None:
        clip_indices = np.where((wlen >= wavelen_range[0]) & (wlen <= wavelen_range[1]) & ~np.isnan(flux))
        wlen = wlen[clip_indices]
        flux = flux[clip_indices]
        errors = errors[clip_indices]
    
    return CRIRESEchelleSpectrum(
        parameters={"filter_type": filter_type, "filter_size": filter_size, "data_file": data_file, "wavelength_file": wavelength_file, "bp_sigma":bp_sigma,"masked_ranges":masked_ranges},
        wlen=wlen,
        flux=flux,
        errors=errors,
        sigma_clip=sigma_clip,
        remove_continuum=remove_continuum,
        filter_type=filter_type,
        filter_size=filter_size,
        bp_sigma=bp_sigma,
        masked_ranges=masked_ranges,
        metadata=metadata,
    )


class CRIRESEchelleDataLoader(DataLoader[CRIRESOrder]):
    """
    A provider for loading CRIRES+ spectral data and doing some processing.
    Assumes a data directory with sub-dirs for each target, and in each a subdir for each night that is structured 
    like excalibuhr output. Produces a dataset of CRIRESOrder objects (NOT CRIRESEchelleSpectrum) (can be selected by order + night)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @property
    def provided_parameters(self) -> Dict[str, List[Any]]:
        return {"target": []}

    def _call(self, parameters={}) -> Dataset[CRIRESOrder]:
        target_name = parameters.get("target", self.config["target"])
        target_cfg = self.config[target_name]
        data_cfg = target_cfg["data"]
        merged_cfg = dict(data_cfg).copy()
        merged_cfg.update(parameters)
        
        remove_continuum = merged_cfg.get('remove_continuum',True)
        sigma_clip = merged_cfg.get('sigma_clip',True)
        
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
        nights = merged_cfg['nights']
        
        disp_name = parameters.get("display_name",target_cfg["display_name"])
        for n in nights:
            datadir = join(basedir,n,"out","combined")
            assert exists(datadir), f"Data directory {datadir} does not exist."
            data_file = join(datadir, data_filename)
            wavelength_file = join(datadir, wavelength_filename)
            self.write_out(f"Loading data for target {target_name} from night {n} in directory {datadir} over wavelength range {wavelen_range}")
            metadata = dict(target=target_name,display_name=disp_name,night=n,directory=datadir, datafile=data_filename,wavelength_filename=wavelength_filename)
            s = load_crires_echelle_spectrum(
                data_file, wavelength_file, wavelen_range, sigma_clip, remove_continuum, filter_size, filter_type, bp_sigma, masked_ranges, metadata=metadata
            )
            spectra.extend(s.spectra)  # grab the orders, discard the rest
        
        dims = []
        if len(nights) > 1:
            dims.append("night")
        if len(spectra) > len(nights):
            dims.append("order") 
        
        # dims = ["night"]
        
        # dims = []
        # if len(merged_cfg["nights"]) > 1:
        #     dims.append("night")

        return Dataset(spectra, dims=dims)