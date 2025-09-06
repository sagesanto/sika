import matplotlib.pyplot as plt
import numpy as np

from sika.implementations.spectroscopy.crires.crires_spectrum import CRIRESSpectrum
from sika.implementations.spectroscopy.spectra.spectrum import (
    Spectrum
)

def plot_crires_model(
    model: Spectrum,
    data: CRIRESSpectrum,
    spectral_order: int,
    selector: dict,
    fig=None,
    axes=None,
):
    import matplotlib.ticker as pltticker

    order_wlen = data.wlen[spectral_order]
    wlen_range = [min(order_wlen), max(order_wlen)]

    data_flux = data.flux[spectral_order]

    mask = np.isin(model.wlen, order_wlen)

    model_wlen = model.wlen[mask]
    model_flux = model.flux[mask]
    # this is the data error multiplied by an error inflation term (based on goodness of scale factor optimization)
    model_errors = model.errors[mask]
    residuals = model.metadata["residuals"][mask]
    resid_sigma = residuals / model_errors
    chi_sq = np.sum(residuals**2 / model_errors**2)
    dof = len(order_wlen) - model.metadata["n_free_params"]  # degrees of freedom
    reduced_chi_2 = chi_sq / dof

    if fig is None or axes is None:
        fig = plt.figure(constrained_layout=True, figsize=(20, 7))
        axes = fig.subplot_mosaic(
            [["Indiv", "Indiv"], ["Spectra", "Spectra"], ["Residuals", "Residuals"]],
            gridspec_kw={"height_ratios": [1, 2, 1]},
        )

    spec_ax = axes["Spectra"]
    res_ax = axes["Residuals"]
    indiv_ax = axes["Indiv"]

    selector_str = ", ".join([f"{k} {v}" for k,v in selector.items()])

    NORM_ADD_FACTOR = 1-np.mean(data_flux)

# --------- Plot individual models -------
    rename_dict = model.metadata.get("model_disp_names",{})
    for k, v in model.metadata["models"].items():
        name = v["dispname"]
        indiv_ax.plot(model_wlen, v["flux"][spectral_order]+NORM_ADD_FACTOR, label=name)

    indiv_ax.set_title(
        f"Reduced $\chi^2$={reduced_chi_2:.4f}, order {spectral_order}, {selector_str}"
    )
    indiv_ax.set_xlim(*wlen_range)
    indiv_ax.set_xticks([])
    indiv_ax.set_ylabel("Normalized flux", fontsize=12)
    indiv_ax.legend(ncols=len(model.metadata["models"]),loc="lower right")

# -------- Plot Model vs Data -----------
    target_name = data.metadata.get("display_name",data.metadata.get("target","Data"))
    spec_ax.plot(order_wlen, data_flux+NORM_ADD_FACTOR, label=target_name, color="black", alpha=0.75,)

    spec_ax.plot(
        model_wlen,
        model_flux+NORM_ADD_FACTOR,
        linestyle="dashed",
        color="red",
        label="Full model",
    )
    # spec_ax.plot(order_wlen, model_flux, alpha=0.75,linestyle="dashed",color="red")
    spec_ax.set_xlim(*wlen_range)
    # spec_ax.set_xlabel("Wavelength (microns)")
    spec_ax.legend()
    spec_ax.legend(ncols=2,loc="lower right")
    spec_ax.set_ylabel("Normalized flux",fontsize=12)

# -------- Plot residuals -----------
    res_ax.plot(model_wlen, resid_sigma, color="gray")
    res_ax.axhline(0, color="red", linestyle="dashed")
    res_ax.set_xlim(*wlen_range)
    res_ax.set_ylim(-5,5)
    res_ax.set_xlabel("Wavelength (microns)",fontsize=12)
    res_ax.set_ylabel("Residuals ($\sigma$)",fontsize=12)

    res_ax.minorticks_on()
    major_loc = pltticker.MultipleLocator(base=0.001)
    minor_loc = pltticker.MultipleLocator(base=0.00025)
    res_ax.xaxis.set_major_locator(major_loc)
    res_ax.xaxis.set_minor_locator(minor_loc)
    res_ax.tick_params(
        axis="x",
        which="major",
        color="k",
        width=2,
        length=6,
        direction="in",
    )
    res_ax.tick_params(
        axis="x",
        which="minor",
        color="k",
        width=1,
        length=6,
        direction="in",
    )

    fig.align_ylabels()

    fig.suptitle(target_name, fontsize=16)
    return fig, axes