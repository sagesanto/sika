from os import listdir
from os.path import join, exists, basename, splitext
from typing import List
from datetime import datetime

from scipy.ndimage import gaussian_filter
from astropy import stats
from astropy.time import Time
from astropy.coordinates import SkyCoord, angular_separation, FK5
import astropy.constants as consts
import astropy.units as u
import scipy.ndimage as ndi
import pytz
from pytz import UTC as dtUTC
import numpy as np
import matplotlib.lines as mlines
import pandas as pd
import tqdm

from sika.utils import parse_path, write_out
from sika.config import Config
from sika.modeling import Dataset

def jd_to_dt(hjd):
    time = Time(hjd, format='jd', scale='tdb')
    return time.to_datetime().replace(tzinfo=pytz.UTC)

def dt_to_jd(datetime):
    return Time(datetime).jd

fk5_apparent = FK5(equinox=Time(datetime.now(tz=dtUTC).year,format="jyear"))

def ang_sep(c1, c2): return angular_separation(c1.ra, c1.dec, c2.ra, c2.dec)

def J2000_to_apparent(coord):
    # technically this actually goes fk5 -> fk5now but should be pretty close?
    c = SkyCoord(coord.ra, coord.dec, frame='fk5')
    return c.transform_to(fk5_apparent)


def integrate_flux(wlen,flux,min_wlen,max_wlen):
    band_inds = np.where((wlen>min_wlen) & (wlen<max_wlen))
    norm_width = max_wlen - min_wlen
    # integrate over bandpass, remembering to normalize by width. Need to integrate the flux in per-micron units
    # assume filter transmission is constant - shouldn't matter much here as we will only use this value for calculating relative flux.
    return np.trapz(flux[band_inds], wlen[band_inds]) / norm_width

# by jerry xuan
def convolve_to_instrument_res(input_wavelength, input_flux, instrument_res):
        """
        This function convolves a HR model spectrum to the instrumental resolution (lower)
        using the provided data_resolution
        Args:
            input_wavelength : numpy.ndarray
                The wavelength grid of the HR model spectrum
            input_flux : numpy.ndarray
                The flux as computed by the model
            instrument_res : float
                :math:`\\lambda/\\Delta \\lambda`, the width of the gaussian kernel to convolve with the model spectrum.
        Returns:
            flux_LSF
                The convolved spectrum.
        """

        # From talking to Ignas: delta lambda of resolution element
        # is FWHM of the LSF's standard deviation, hence:
        sigma_LSF = 1./instrument_res/(2.*np.sqrt(2.*np.log(2.)))

        # The input spacing of petitRADTRANS is 1e3, but just compute
        # it to be sure, or more versatile in the future.
        # Also, we have a log-spaced grid, so the spacing is constant
        # as a function of wavelength
        spacing = np.mean(2.*np.diff(input_wavelength)/ \
                          (input_wavelength[1:]+input_wavelength[:-1]))

        # Calculate the sigma to be used in the gauss filter in units
        # of input wavelength bins
        sigma_LSF_gauss_filter = sigma_LSF/spacing

        flux_LSF = gaussian_filter(input_flux, sigma = sigma_LSF_gauss_filter,  mode = 'nearest')

        return flux_LSF


def _make_pressure_grid(tier_csv_path, config):
    """ Makes a csv file that separates out the elf owl model files by the pressure grids that they employ. this can be used to sort models into groups that can share a common pressure grid, speeding up the pRT process"""
    import xarray
    bounds = []
    params = []
    species = None

    elf_dir = parse_path(config["elf_owl"]["model_dir"])
    for f in tqdm.tqdm(listdir(elf_dir)):
        if species is None:
            ds = xarray.load_dataset(join(elf_dir,f))
            cols = list(ds._variables.keys())
            species = [s for s in cols if s not in ("temperature","flux","pressure","wavelength","e-")]
            ds.close()
        ds = xarray.open_dataset(join(elf_dir,f),drop_variables=species)
        p = np.array(ds["pressure"].values)
        bounds.append(np.array([np.min(p),np.max(p)]))
        fname = splitext(basename(f))[0]
        parts = fname.split("_")[1:]
        parameters = {parts[i]: float(parts[i + 1]) for i in range(0, len(parts), 2)}
        params.append(parameters)
        ds.close()
    bounds = np.array(bounds)
    logbounds = np.log10(bounds)
    rounded_bounds = np.trunc(logbounds*10)/10.0
    unique_bounds = np.unique(rounded_bounds, axis=0)
    indices = np.searchsorted(unique_bounds[:,1],rounded_bounds)[:,1]
    rows = []

    for (p, rbounds, classification) in zip(params, rounded_bounds, indices):
        row = {}
        row["start"], row["stop"] = rbounds
        row.update(p)
        row["logkzz"] = row.pop("logzz")
        row["class"] = classification

        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(tier_csv_path, index=False)


def pressure_grid_tiers(config, logger):
    """ Makes or reads a csv file and df that separates out the elf owl model files by the pressure grids that they employ. this can be used to sort models into groups that can share a common pressure grid, speeding up the pRT process"""
    tier_csv_path = join(parse_path(config["filestore"]), "eo_pressure_grid_bounds.csv")
    if not exists(tier_csv_path):
        write_out("Figuring out pressure grid tiers...",logger=logger)
        _make_pressure_grid(tier_csv_path, config)
        write_out("Done!",logger=logger)
    grid_tiers = pd.read_csv(tier_csv_path)
    grid_tiers.sort_values(by=["stop", "start"], inplace=True)
    grid_tiers.reset_index(drop=True, inplace=True)

    class_groups = [group for _, group in grid_tiers.groupby("class")]
    g = []
    write_out(f"Found {len(class_groups)} pressure grid tiers", logger=logger)
    for i, group in enumerate(class_groups):
        param_list = group[["teff","grav","mh","co","logkzz"]].to_dict(orient="records")
        g.append((group.iloc[0]["start"], group.iloc[0]["stop"], param_list))
        write_out(f"Group {i+1}: {len(param_list)} models with pressure grid from {group.iloc[0]['start']} to {group.iloc[0]['stop']}", logger=logger)
    return g


def plot_spectra(wlen, flux, ax, label, xlim=None, ylim=None, **kwargs):
    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    default_kwargs = {
        "alpha":0.5
    }
    default_kwargs.update(kwargs)
    merged_kwargs = default_kwargs

    ax.plot(wlen, flux, label=label, **merged_kwargs)

    ax.set_ylabel(r"Flux [erg/s/cm$^2$/cm]", fontsize=12)
    ax.set_xlabel("Wavelength [microns]", fontsize=12)

    ax.minorticks_on()
    ax.tick_params(
        axis="both",
        which="major",
        color="k",
        length=18,
        width=2,
        direction="in",
        labelsize=16,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        color="k",
        length=12,
        width=1,
        direction="in",
        labelsize=16,
    )

# by jerry xuan
# adolfo function for rotational broadening - works much faster than rotBroad and is good for any range of wavelength! 
def rot_int_cmj(wlen:np.ndarray, flux:np.ndarray, vsini:float, eps:float=0, nr:int=10, ntheta:int=100, dif:float=0.0) -> np.ndarray:
    """
    adolfo function for rotational broadening - works much faster than rotBroad and is good for any range of wavelength!

    :param wlen: wavelength array (microns)
    :param flux: flux array (arbitrary units)
    :param vsini: rotational velocity in km/s
    :param eps: limb darkening parameter (default is 0, no limb darkening)

    :returns: flux array, rotationally broadened
    """
    if vsini == 0 and eps == 0:
        return flux
    
    ns = np.copy(flux)*0.0
    tarea = 0.0
    dr=1./nr
    for j in range(0, nr):
        r = dr/2.+j*dr
        area=((r+dr/2.)**2-(r-dr/2.)**2)/int(ntheta*r)*(1.-eps+eps*np.cos(np.arcsin(r)))
        for k in range(0,int(ntheta*r)):
            th = np.pi/int(ntheta*r)+k*2.*np.pi/int(ntheta*r)
            if dif != 0:
                vl=vsini*r*np.sin(th)*(1.-dif/2.-dif/2.*np.cos(2.*np.arccos(r*np.cos(th))))
                ns=ns+area*np.interp(wlen+wlen*vl/2.9979e5,wlen, flux)
                tarea=tarea+area
            else:
                vl=r*vsini*np.sin(th)
                ns=ns+area*np.interp(wlen+wlen*vl/2.9979e5,wlen,flux)
                tarea=tarea+area
    ns = ns/tarea
    return ns

# adapted from jerry xuan
def apply_rv_shift(rv: float, wlen: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """apply rv shift to a spectrum

    :param rv: radial velocity in km/s
    :type rv: float
    :param wlen: wavelength in microns
    :type wlen: np.ndarray
    :param flux: flux in arbitrary units
    :type flux: np.ndarray
    :return: flux shifted to the new wavelength
    :rtype: np.ndarray
    """
    new_beta = rv / consts.c.to(u.km/u.s).value
    new_redshift = np.sqrt((1 + new_beta)/(1 - new_beta)) - 1
    template_wvs_shifted = wlen/(1+new_redshift)
    return np.interp(template_wvs_shifted, wlen, flux)

# adapted from jerry xuan
def optimize_scale_factors(data_flux, data_error, model_fluxes: List[np.ndarray]):
    """ optimize scale factors to find the best-fit linear combination of model fluxes to match the data fluxes """
    scale_factors = []
    ncomp = len(model_fluxes)
    # print(len(np.where(np.isnan(data_flux))[0]),"NaNs in data_flux")
    # print(len(np.where(np.isnan(data_error))[0]),"NaNs in data_error")
    # print(len(np.where(np.isnan(model_fluxes))[0]),"NaNs in model_fluxes")
    # # how to construct linear model for two component model?
    # print("n models:", ncomp)
    # print("data_flux shape:", data_flux.shape)
    # print("data_error shape:", data_error.shape)
    # print("model_fluxes:", model_fluxes)
    
    y_model = np.array(model_fluxes).T
    # print("y_model shape:", y_model.shape)
    invalid = False
    if y_model.shape[0] == 0:
        print(f"WARNING: trying to optimize scale factors but the model fluxes have zero length! (shape {y_model.shape})")
        invalid = True
    if data_flux.shape[0] == 0:
        print(f"WARNING: trying to optimize scale factors but the data flux has zero length! (shape {data_flux.shape})")
        invalid = True
    if data_error.shape[0] == 0:
        print(f"WARNING: trying to optimize scale factors but the data error has zero length! (shape {data_error.shape})")
        invalid = True
    if invalid:
        return [0.0]*ncomp, np.inf
    #cov = np.array([comp.eflux[d][c]**2 + 10**this_emult, comp.eflux[d][c]**2 + 10**this_emult]).T
    cov = np.array([data_error**2]*ncomp).T
    # print("cov shape:", cov.shape)
    # cov = np.array([comp.eflux[d][c]**2, comp.eflux[d][c]**2]).T
    # y_data = np.array([comp.norm_flux[d][c], comp.norm_flux[d][c]]).T
    y_data = np.array([data_flux]*ncomp).T
    # print("y_data shape:", y_data.shape)
    #print(y_model.shape, cov.shape, y_data.shape)
    lhs = np.dot(y_model.T, 1/cov * y_model)
    rhs = np.dot(y_model.T, 1/cov * y_data)
    if y_model.ndim == 1:
        f_det = rhs/lhs
    else:
        f_det = np.linalg.solve(lhs, rhs)

    #print(f_det.shape, f_det)
    scale_factors = f_det[:,0]
    # Ba_scale = f_det[0, 0]
    # Bb_scale = f_det[1, 0]
    #print(Ba_scale, Bb_scale)
    # construct model from scale factors
    # this_final_model = flux_model_Rdata * Ba_scale + flux_model_Rdata2 * Bb_scale
    final_model = np.sum([model_fluxes[i] * scale_factors[i] for i in range(ncomp)], axis=0)
    # # compute error multiple factor directly
    # chi_squared = np.sum(((comp.norm_flux[d][c]-this_final_model)**2/comp.eflux[d][c]**2))
    chi_squared = np.sum(((data_flux-final_model)**2/data_error**2))
    beta = np.sqrt(chi_squared/len(data_flux))
    # beta = np.sqrt(chi_squared/len(comp.norm_flux[d][c]))
    return scale_factors, beta

# adapted from jerry xuan
# filt_type -> filter_type: can be 'median' or 'gaussian'
# medfilt -> filter_size: size of the median filter to apply
def clean_and_normalize_spectrum(fluxes, wavelengths, errors, bp_sigma=3, filter_type='median', filter_size=100):
    clipped_indices = stats.sigma_clip(fluxes, sigma=bp_sigma)
    spike_inds = np.where(clipped_indices.mask==True)
    # print(f"deleting {sum(len(i) for i in spike_inds)} spike points")
    # plt.scatter(wavelengths[spike_inds], fluxes[spike_inds], s=1, color='red', label='bad pixels')
    # plt.show()
    # delete the nans and bad pixels
    wavelengths = np.delete(wavelengths, spike_inds)
    fluxes = np.delete(fluxes, spike_inds)
    errors = np.delete(errors, spike_inds)
    if filter_type == 'median':
        continuum = ndi.median_filter(fluxes, filter_size)
    elif filter_type == 'gaussian':
        continuum = ndi.gaussian_filter(fluxes, filter_size)
    else:
        raise ValueError("filter_type must be 'median' or 'gaussian'")

    norm_constant = np.nanmedian(continuum)
    # May 21, 2024 - do not add the constant factor back anymore 
    fluxes = fluxes - continuum
    return fluxes, wavelengths, errors, norm_constant

class KBandLossAdjustment:
    """Contributes a penalty term for deviation from the observed k-band flux to the loss of a binary model. This is very simple to implement and pretty specific in its expectation of the metadata structure of the model. It's probably easier to write a new one for your specific use case than to try to conform to this one."""
    def __init__(self, obs_kband_ratio: float, kband_ratio_error: float):
        self.obs_kband_ratio = obs_kband_ratio
        self.kband_ratio_error = kband_ratio_error

    def __call__(self, loss: float, parameters: List[float], model: Dataset, data: Dataset, errors: np.ndarray, residuals: np.ndarray, config: Config):
        # grab the first model - i'm assuming all models will exhibit the same k-band ratio
        first_model = model.values(model.selectors[0])
        
        assert "models" in first_model.metadata, "KBandLossAdjustment requires that the model dataset has 'models' metadata containing metadata about the individual component models."
        k_fluxes = []
        for k, v in first_model.metadata["models"].items():
            if 'metadata' not in v:
                continue
            v = v['metadata']
            if "models" not in v:
                continue
            for k2, v2 in v["models"].items():
                if 'metadata' not in v2:
                    continue
                v2 = v2['metadata']
                if "k_band_flux" not in v2:
                    continue
                k_fluxes.append(v2["k_band_flux"])

        if len(k_fluxes) != 2:
            raise ValueError(f"KBandLossAdjustment requires exactly two components to compute the k-band flux ratio, but found {len(k_fluxes)} components.")
        
        k_ratio = k_fluxes[1] / k_fluxes[0]
        
        k_ratio_resid = self.obs_kband_ratio - k_ratio
        all_chi2 = np.nansum((k_ratio_resid ** 2 / self.kband_ratio_error ** 2))

        return -0.5 * all_chi2

class SpecificKBandLossAdjustment:
    """Contributes a penalty term for deviation from the observed k-band flux to the loss of a binary model.
    Ratio is defined as model_2 / model_1.
    
    This is very simple to implement and pretty specific in its expectation of the metadata structure of the model, so it's probably easier to write a new one for your specific use case than to try to conform to this one."""
    def __init__(self, model_1_name, model_2_name, obs_kband_ratio: float, kband_ratio_error: float):
        self.obs_kband_ratio = obs_kband_ratio
        self.model_1_name = model_1_name
        self.model_2_name = model_2_name
        self.kband_ratio_error = kband_ratio_error

    def __call__(self, loss: float, parameters: List[float], model: Dataset, data: Dataset, errors: np.ndarray, residuals: np.ndarray, config: Config):
        # grab the first model - i'm assuming all models will exhibit the same k-band ratio
        
        first_model = model.values(model.selectors[0])
        
        assert "models" in first_model.metadata, "KBandLossAdjustment requires that the model dataset has 'models' metadata containing metadata about the individual component models."
        
        models_meta = first_model.metadata["models"]
        
        if self.model_1_name not in first_model.metadata['models']:
            raise ValueError(f"Couldn't find model '{self.model_1_name}' in model metadata. Known models are: {first_model.metadata['models']}")
        if self.model_2_name not in first_model.metadata['models']:
            raise ValueError(f"Couldn't find model '{self.model_2_name}' in model metadata. Known models are: {first_model.metadata['models']}")
        
        
        try: 
            model_1_kband = models_meta[self.model_1_name]["metadata"]["k_band_flux"]
        except KeyError:
            raise ValueError(f"Couldn't find key 'k_band_flux' in metadata of model '{self.model_1_name}'. Model metadata: {models_meta[self.model_1_name]['metadata']}")
        
        try:
            model_2_kband = models_meta[self.model_2_name]["metadata"]["k_band_flux"]
        except KeyError:
            raise ValueError(f"Couldn't find key 'k_band_flux' in metadata of model '{self.model_2_name}'. Model metadata: {models_meta[self.model_2_name]['metadata']}")
        
        k_ratio = model_2_kband / model_1_kband
                
        k_ratio_resid = self.obs_kband_ratio - k_ratio
        all_chi2 = np.nansum((k_ratio_resid ** 2 / self.kband_ratio_error ** 2))

        return -0.5 * all_chi2