import numpy as np
import requests
import os
from astropy.io import fits
import warnings
from io import BytesIO

from sika.provider import Provider
from sika.implementations.spectroscopy import Spectrum

__all__ = ["Phoenix","download_PHOENIX_stellar_model"]

def download_PHOENIX_stellar_model(teff, logg, url='https://phoenix.astro.physik.uni-goettingen.de/data/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/'):
    """
    Get the PHOENIX stellar model.

    Parameters
    ----------
    teff: int
        The effective temperature of the standard star.
    wave_cut: list
        The wavelength range to use for the stellar model.
    logg: float
        The surface gravity of the standard star.

    Returns
    -------
    w: array
        The wavelength of the stellar model.
    f: array
        The flux of the stellar model.
    """
    
    if teff < 2300 or teff > 12000:
        raise ValueError(f"Telluric temperature {teff} K is out of Phoenix model range (2300-12000 K).")
    
    modulus = 100 if teff < 7000 else 200
    if teff % modulus != 0:
        raise ValueError(f"Telluric temperature {teff} K is not a multiple of {modulus} K!")

    filename = f'lte{teff:05d}-{logg:.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
    # if not os.path.isfile(f"models/{filename}"):
    print(f"Downloading PHOENIX stellar model T={teff} K")
    r = requests.get(url+filename, timeout=10)
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise RuntimeError(f"Failed to download a Phoenix stellar model for temp={teff}, logg={logg} ({url}): {err}") from err

    # with open(f"models/{filename}", "wb") as f:
    #     f.write(r.content)
    # print("[DONE]")
    # flux = fits.getdata(f"models/{filename}")  #'erg/s/cm^2/cm'
    bytestream = BytesIO(r.content)
    flux = fits.getdata(bytestream)  #'erg/s/cm^2/cm'
    f = flux/np.median(flux)
    return f


class Phoenix(Provider):
    def _setup(self):
        self.url = self.config["phoenix"]["url"]
        self.model_dir = self.config["phoenix"]["model_dir"]
        self.wave_file = self.config["phoenix"]["wave_file"]
        self.wlen = fits.getdata(self.wave_file) / 1e4 # to micron
    
    @property
    def provided_parameters(self):
        return {
            "teff": np.concatenate(
                (np.arange(2300, 7000, 100),
                 np.arange(7000, 12000, 200))),
            "logg": np.arange(0.0,6.0,0.5)
        }
        
    def _call(self, parameters):
        teff, logg = parameters["teff"], parameters["logg"]
        if teff not in self.provided_parameters["teff"]:
            raise ValueError(f"Invalid teff: {teff}. Valid teff values are {self.provided_parameters['teff']}.")
        if logg not in self.provided_parameters["logg"]:
            raise ValueError(f"Invalid logg: {logg}. Valid logg values are {self.provided_parameters['logg']}.")
        
        flux = download_PHOENIX_stellar_model(teff, logg, url=self.url)
        spec = Spectrum(parameters={'teff':teff,'logg':logg},
                           flux = flux,
                           wlen=self.wlen,
                           metadata={'model_name':'phoenix', 'wave_file':self.wave_file}
                )
        return spec