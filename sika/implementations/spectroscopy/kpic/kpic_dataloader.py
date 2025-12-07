from os.path import join, exists
from glob import glob
from typing import Dict, List, Any, Tuple
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


from sika.modeling import Dataset, DataLoader
from sika.implementations.spectroscopy.kpic.kpic_spectrum import KPICSpectrum, KPICOrder
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


# def load_crires_spectrum(data_file: str, wavelength_file: str, wavelen_range: tuple, filter_size:int, filter_type:str, bp_sigma:int, masked_ranges:List[Tuple[int,int]]=None) -> CRIRESSpectrum:
def load_kpic_spectrum(
    data_dir,
    calib_dir,
    exposures: list[str],
    fiber: str,
    is_star: bool,
    filter_size: int,
    filter_type: str,
    bp_sigma: int,
    masked_ranges: List[Tuple[int, int]] = None,
    orders=None,
    response_file=None,
    metadata=None
) -> KPICSpectrum:
    """Load a KPIC spectrum from file.

    :param data_dir: Path to target's science dir for this night
    :type data_dir: str
    :param calib_dir: Path to calibration dir for this night
    :type calib_dir: str
    :param exposures: List of exposure filenames to include
    :type exposures: list[str]
    :param filter_size: Size of the filter to apply
    :type filter_size: int
    :param filter_type: Type of the filter to apply - 'median' or 'gaussian'
    :type filter_type: str
    :rtype: CRIRESSpectrum
    """

    # determine which frames will be combined for this dataset
    # data_dir = join(data_dir,target_name,night)
    assert exists(data_dir)
    # print(data_dir)
    metadata = metadata or {}
    
    # find the flux files for the requested exposures (if any)
    if exposures is None:
        filelist = glob(join(data_dir, "fluxes", "*fits"))
    else:
        filelist = [s for e in exposures for s in glob(join(data_dir, "fluxes", f"*{e}*fits"))]
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
        masked_ranges=masked_ranges,
        orders=orders,
        response_wlen=response_wlen,
        response_flux=response_flux,
        metadata=metadata
    )


class KPICDataLoader(DataLoader[KPICOrder]):
    @property
    def provided_parameters(self) -> Dict[str, List[Any]]:
        return {"target": []}

    def _call(self, parameters={}) -> Dataset[KPICOrder]:
        target_name = parameters.get("target", self.config["target"])
        target_cfg = self.config[target_name]
        data_cfg = target_cfg["data"]
        merged_cfg = dict(data_cfg).copy()
        merged_cfg.update(parameters)
        
        
        # wavelen_range = merged_cfg["wavelen_range"]
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
            if not is_star:
                response_file = join(data_dir, data_cfg[n]["response_file"])
            else:
                response_file = None
            exposure_num_cfg = merged_cfg[n]["exposures"]
            for fiber in fibers:
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
                    metadata=metadata
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