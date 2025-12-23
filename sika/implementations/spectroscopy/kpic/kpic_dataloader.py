from logging import Logger
from os.path import join, exists
from glob import glob
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum
import numpy as np
import pandas as pd
import warnings

from astropy import stats

import kpicdrp
import kpicdrp.data as drp_data
from kpicdrp import utils as drp_utils
from kpicdrp.caldb import wave_caldb, trace_caldb
from kpicdrp.data import BadPixelMap, Background, TraceParams, DetectorFrame, Wavecal, Spectrum as KSpectrum


from sika.config.config import Config
from sika.modeling import Dataset, DataLoader
from sika.implementations.spectroscopy.kpic.kpic_spectrum import KPICSpectrum, KPICOrder
from sika.task import Task
from sika.utils import parse_path

warnings.filterwarnings("ignore",module="astropy.stats.sigma_clipping")
__all__ = ["load_kpic_spectrum", "KPICDataLoader"]

class CalibType(Enum):
    WAVECAL = "*wvs.fits"
    TRACE = "*trace.fits"


def combine_flux(flux_list, e_flux_list, sigma=5, err_comb_method="mean"):
    # from Jerry Xuan

    assert len(flux_list) == len(e_flux_list)

    # print(type(e_flux_list), e_flux_list.shape)

    flux = np.nanmean(
        stats.sigma_clip(flux_list, sigma=sigma, axis=0, masked=False), axis=0
    )
    # median more robust to outliers
    if err_comb_method == "mean":
        e_flux = np.nanmean(
            stats.sigma_clip(e_flux_list, sigma=sigma, axis=0, masked=False), axis=0
        ) / np.sqrt(len(e_flux_list))

        # print(e_flux_list[0][6][500:550], e_flux_list[1][6][500:550] )
        # print(e_flux.shape)
        # print(e_flux[6][500:550], np.nanmedian(e_flux[6]))

    elif err_comb_method == "median":
        e_flux = np.nanmedian(
            stats.sigma_clip(e_flux_list, sigma=sigma, axis=0, masked=False), axis=0
        ) / np.sqrt(len(e_flux_list))

    return np.array(flux), np.array(e_flux)


def find_cal_file(calib_dir, calib_type: CalibType):
    filelist = glob(join(calib_dir, calib_type.value))
    if len(filelist) > 1:
        raise ValueError(
            f"Expecting only one {calib_type.value} file in the calib directory {calib_dir}, instead found {len(filelist)}: {filelist}"
        )
    try:
        file = filelist[0]
    except IndexError:
        raise FileNotFoundError(
            f"Could not find a {calib_type.value} file in directory {calib_dir}."
        )
    return file


def load_wavecal_manual(calib_dir):
    file = find_cal_file(calib_dir, CalibType.WAVECAL)
    return Wavecal(filepath=file)


def load_trace_manual(calib_dir):
    file = find_cal_file(calib_dir, CalibType.TRACE)
    return TraceParams(filepath=file)


def combine_planet_spectrum(planet_specs: drp_data.Dataset, fiber: str):
    # these have shape (nframes, ntrace, norders, 2048)
    all_trace_flux = np.array(planet_specs.get_dataset_attributes("fluxes"))
    all_trace_e_flux = np.array(planet_specs.get_dataset_attributes("errs"))

    # determine where the data for our fiber is
    science_fiber_indices = planet_specs[0].trace_index
    sci_fiber_idx = science_fiber_indices[fiber.lower()]

    # select the flux and err for our fiber
    all_flux = all_trace_flux[:, sci_fiber_idx, :, :]
    all_e_flux = all_trace_e_flux[:, sci_fiber_idx, :, :]

    # print(all_flux.shape)

    # combine the fluxes from the different frames
    err_comb = "mean"
    flux, err = combine_flux(all_flux, all_e_flux, sigma=5, err_comb_method=err_comb)
    return flux, err


def combine_star_spectrum(star_specs: drp_data.Dataset, fiber: str):
    combined_star_spec = kpicdrp.utils.stellar_spectra_from_files(star_specs)
    fiber_index = combined_star_spec.trace_index[fiber.lower()]
    flux = combined_star_spec.fluxes[
        fiber_index
    ]  # should always be 0, since I only load files for 1 single fiber
    e_flux = combined_star_spec.errs[fiber_index]
    return flux, e_flux


def load_kpic_spectrum(
    data_dir: str,
    calib_dir: str,
    exposures: list[str],
    fiber: str,
    is_star: bool,
    filter_size: int,
    filter_type: str,
    bp_sigma: int,
    masked_ranges: Optional[List[Tuple[float, float]]] = None,
    normalize: bool = False,
    orders: Optional[List[int]] = None,
    response_file:Optional[str] = None,
    metadata:Optional[Dict[str,Any]] = None,
    flux_dir_extension:Optional[str] = 'fluxes'
) -> KPICSpectrum:
    """Load a KPIC spectrum from file.

    :param data_dir: Path to target's science dir for this night
    :type data_dir: str
    :param calib_dir: Path to calibration dir for this night
    :type calib_dir: str
    :param exposures: List of exposure filenames to include
    :type exposures: list[str]
    :param fiber: the name of the fiber to extract spectrum from. ex. 's2'. To load multiple fibers from the same file, call this function multiple times.
    :type fiber: str
    :param is_star: whether or not the data is a spectrum of a star, which changes whether it will be loaded with :py:meth:`~sika.implementations.spectroscopy.kpic.kpic_dataloader.combine_planet_spectrum` or :py:meth:`~sika.implementations.spectroscopy.kpic.kpic_dataloader.combine_star_spectrum` behind the scenes. (see the kpicdrp `stellar_spectra_from_files <https://github.com/kpicteam/kpic_pipeline/blob/main/kpicdrp/utils.py#L128/>`__ for more)
    :param filter_size: Size of the filter to apply
    :type filter_size: int
    :param filter_type: Type of the filter to apply - 'median' or 'gaussian'
    :type filter_type: str
    :param bp_sigma: the sigma at which flux should be sigma-clipped
    :type bp_sigma: float
    :param masked_ranges: a list of (wlen_start, wlen_end) tuples that denote ranges of wavelengths that should be masked in the spectrum during processing and then deleted
    :type masked_ranges: List[Tuple[float, float]]
    :param normalize: whether to normalize each order's flux by dividing by its 90th percentile value. default False
    :type normalize: bool
    :param orders: if provided, will keep only the orders specified in this **zero-index** list of indices. loads and keeps all orders by default.
    :type orders: List[int] | None
    :param response_file: the path to a KPIC response file containing a kpicdrp.drp_data.Spectrum response for the targeted fiber. required if this spectra will be used as data input to :py:class:`~sika.implementations.spectroscopy.kpic.kpic_model.KPICModel`
    :type response_file: str
    :param metadata: a dictionary of metadata to attach to the loaded spectrum. spectra loaded by this function will always additionally have 'fiber', 'exposures', 'is_star', 'response_file', and 'data_dir' metadata attached
    :type metadata: Optional[dict[str,Any]]
    :param flux_dir_extension: the name of the directory immediately above the exposure files. defaults to 'fluxes'. this is NOT the full path to the flux file directory!
    :type flux_dir_extension: Optional[str]
    :rtype: KPICSpectrum
    """

    # determine which frames will be combined for this dataset
    # data_dir = join(data_dir,target_name,night)
    assert exists(data_dir)
    # print(data_dir)
    metadata = metadata or {}
    
    # find the flux files for the requested exposures (if any)
    if exposures is None:
        filelist = glob(join(data_dir, flux_dir_extension, "*fits"))
    else:
        filelist = [s for e in exposures for s in glob(join(data_dir, flux_dir_extension, f"*{e}*fits"))]
    # print(filelist)
    
    # load the dataset as a raw science dataset in order to later use the caldb
    raw_sci_dataset = drp_data.Dataset(filelist=filelist, dtype=drp_data.DetectorFrame)
        
    # load the fluxes as a list of nframes spectra
    specs = drp_data.Dataset(filelist=filelist, dtype=drp_data.Spectrum)

    if is_star:
        flux, err = combine_star_spectrum(specs, fiber)
    else:
        flux, err = combine_planet_spectrum(specs, fiber)

    # load line-spread-function from trace file
    try:
        trace_dat = trace_caldb.get_calib(raw_sci_dataset[0])
    except Exception as e:
        trace_dat = load_trace_manual(calib_dir)
    trace_ind = trace_dat.trace_index[fiber.lower()]
    trace_sigmas = trace_dat.widths[trace_ind]

    # load wavecal
    try:
        wavecal_dat = wave_caldb.get_calib(raw_sci_dataset[0])
    except Exception as e:
        wavecal_dat = load_wavecal_manual(calib_dir)

    # get the wavecal's wavelength arrays
    wvs_ind = wavecal_dat.trace_index[fiber.lower()]
    waves = wavecal_dat.wvs[wvs_ind]

    if response_file is not None:
        # should load the response here
        spectral_responses = drp_data.Spectrum(filepath=response_file)
        response_fiber_idx = spectral_responses.trace_index[fiber.lower()]
        response_flux = spectral_responses.fluxes[response_fiber_idx, :]
        response_wlen = spectral_responses.wvs[response_fiber_idx, :]
    else:
        response_wlen = None
        response_flux = None

    metadata.update({"fiber":fiber.lower(), "exposures": exposures, "is_star": is_star, "response_file": response_file, "data_dir":data_dir})
    # if orders is not None:
    #     metadata["orders"]=orders

    # the kpic spectrum constructor selects by order and does the filtering
    return KPICSpectrum(
        wlen=waves,
        flux=flux,
        errors=err,
        trace_sigmas=trace_sigmas,
        filter_type=filter_type,
        filter_size=filter_size,
        bp_sigma=bp_sigma,
        normalize=normalize,
        masked_ranges=masked_ranges,
        orders=orders,
        response_wlen=response_wlen,
        response_flux=response_flux,
        metadata=metadata
    )


class KPICDataLoader(DataLoader[KPICOrder]):
    
    def __init__(self, *args, normalize:bool=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize = normalize
    
    @property
    def provided_parameters(self) -> Dict[str, List[Any]]:
        return {"target": []}
    
    # torture
    def find_response_file(self, merged_cfg, target_name, night, fiber):
        kpic_base_science_dir = parse_path(self.config["kpic"]["data_dir"])
        
        explicit_filepath = merged_cfg[night].get("response_file")
        night_data_dir = join(kpic_base_science_dir,target_name, night)
        if explicit_filepath:
            response_file = join(night_data_dir, explicit_filepath)
            if not exists(response_file):
                raise FileNotFoundError(f"Expected to find spectral responses for {target_name} on night {night} and fiber {fiber} in the file {response_file} but couldn't find it. Try confirming that the file is in the correct place, or adjusting the config")
            return response_file

        # if we weren't given an explicit path in the config, we were instead given the name of a response star and we need to go get the correct file
        response_star = merged_cfg.get("response_star")
        if not response_star:
            raise ValueError(f"Tried to load response for target {target_name} on night {night} but couldn't find a 'response_file' or 'response_star' config key - one of the two must be provided.")
        
        response_dir_extension=self.config['kpic'].get('response_directory_name_format',"response")
        response_dir = join(kpic_base_science_dir,response_star,night,response_dir_extension)
        if not exists(response_dir):
            raise FileNotFoundError(f"Expected to find spectral responses for {target_name} on night {night} in the directory {response_dir} but couldn't find it. Try confirming that the files are in the correct place, or adjusting the config")
        response_fname = f"{night}_spec_response_{fiber}.fits" 
        response_file = join(response_dir, response_fname)
        
        # fuckery
        if not exists(response_file) and fiber.startswith('s'):
            fiber_number = fiber[-1]
            response_fname_guess = f"{night}_spec_response_sf{fiber_number}.fits"
            guess_file = join(response_dir, response_fname_guess)
            if not exists(guess_file):
                raise FileNotFoundError(f"Expected to find spectral responses for {target_name} on night {night} and fiber {fiber} in the file {response_file} or {guess_file} but couldn't find either. Try confirming that the file is in the correct place, or adjusting the config")
            response_file = guess_file
            
        elif not exists(response_file):
            raise FileNotFoundError(f"Expected to find spectral responses for {target_name} on night {night} and fiber {fiber} in the file {response_file} but couldn't find it. Try confirming that the file is in the correct place, or adjusting the config")
        
        return response_file
        

    def _call(self, parameters={}) -> Dataset[KPICOrder]:
        target_name = parameters.get("target", self.config["target"])
        target_cfg = self.config[target_name]
        data_cfg = target_cfg["data"]
        merged_cfg = dict(data_cfg).copy()
        merged_cfg.update(parameters)
                
        filter_size = merged_cfg["filter_size"]
        filter_type = merged_cfg["filter_type"]
        orders = merged_cfg.get("orders", None)
        bp_sigma = merged_cfg["bp_sigma"]
        masked_ranges = merged_cfg.get("masked_ranges",[])
        
        # confusing, but: according to Yapeng, the wavelengths should be pulled from the telluric star not the science data
        basedir = parse_path(self.config["kpic"]["data_dir"])
        basedir = join(basedir, target_name)
        calib_basedir = parse_path(self.config["kpic"]["calib_dir"])
        
        # this target is a star if the config does not point to another target as its primary
        is_star = not bool(target_cfg.get("primary", False))
        
        nights = merged_cfg["nights"]
        fibers = merged_cfg["fibers"]
        
        spectra = []
        for n in nights:
            data_dir = join(basedir,n)
            assert exists(data_dir), f"Data directory {data_dir} does not exist."
            calib_dir = join(calib_basedir, n)
            exposure_num_cfg = merged_cfg[n]["exposures"]
            for fiber in fibers:
                if not is_star:
                    response_file = self.find_response_file(merged_cfg, target_name, n, fiber)
                else:
                    response_file = None
                exposures = exposure_num_cfg.get(fiber)
                if exposures is None:
                    raise ValueError(f"Tried to load data for {target_name} fiber '{fiber}' (night {n}) but no exposures for that fiber were specified in the {target_name}.data.{n}.exposures config section.")
                
                self.write_out(f"Loading data for target {target_name} from night {n}, fiber {fiber} in directory {data_dir}. is_star: {is_star}")
                metadata = dict(target=target_name, display_name=parameters.get("display_name",target_cfg["display_name"]),night=n,fiber=fiber.lower(),directory=data_dir)

                s = load_kpic_spectrum(
                    data_dir, 
                    calib_dir, 
                    exposures=exposures,
                    fiber=fiber,
                    is_star=is_star,
                    filter_size=filter_size, 
                    filter_type=filter_type, 
                    bp_sigma=bp_sigma,
                    masked_ranges=masked_ranges,
                    orders=orders,
                    response_file=response_file,
                    metadata=metadata,
                    normalize=self.normalize,
                    flux_dir_extension=self.config['kpic'].get('flux_directory_name_format',"fluxes")
                )
                spectra.extend(s.spectra)
        print(spectra)
        dims = []
        if len(nights) > 1:
            dims.append("night")
        if len(fibers) > 1:
            dims.append("fiber")
        if len(spectra) > len(nights) * len(fibers):
            dims.append("order")       
        print(dims)     
        return Dataset(spectra, dims=dims)