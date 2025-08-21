import sys, os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join, abspath, splitext, basename
import numpy as np

from sika.implementations.spectroscopy import Spectrum
from sika.implementations.spectroscopy.utils import integrate_flux
from sika.config import Config, configure_logger, config_path
from sika.utils import parse_path, savefig, file_timestamp

def parse_filename(filename: str):
    """ Extract model name and parameters from the filename. """
    filename = splitext(basename(filename))[0]  # remove file extension
    parts = filename.split('_')
    parameters = {parts[i]: parts[i + 1] for i in range(0,len(parts), 2)}
    for k,v in parameters.items():
        try:
            parameters[k]=float(v)
        except:
            pass
    return parameters

config = Config(config_path)

dir = join(parse_path(config["filestore"]),  "spectral_models", "pRT", "elf_owl")

rows = []

outdir = parse_path(config["filestore"])
os.makedirs(outdir, exist_ok=True)

for filename in os.listdir(dir):
    if not filename.endswith(".abcmeta"):
        continue
    params = parse_filename(filename)
    # params["filename"] = filename
    spec = Spectrum.load(join(dir,filename))
    flux = spec.flux
    wlen = spec.wlen
    
    k_band_flux = integrate_flux(wlen, flux, 2.025, 2.15)
    params["k_band_flux"] = k_band_flux
    rows.append(params)

df = pd.DataFrame(rows)
df.to_csv(join(outdir, "k_band_fluxes.csv"), index=False)

# plt.title("K-band flux vs. $T_{Eff}$")
# plt.scatter(df["teff"],df["k_band_flux"])
# plt.xlabel("$T_{\text{Eff}}$")
# plt.ylabel("K-band Flux")
# plt.show()