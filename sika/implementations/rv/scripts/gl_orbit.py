import os
import shutil
from os.path import join, exists
import pandas as pd
import numpy as np
import logging
import astropy.units as u

from sika.modeling import Model, Parameter, ParameterSet, PriorTransform, Uniform, Normal, Dataset, NullPriorTransform, AuxiliaryParameterSet
from sika.implementations.rv import RV, BinaryRV, BinaryRVParams, BinaryKeplerianRV, BinaryRVSampler
from sika.config import Config, config_path, configure_logger

from sika.utils import (
    sika_argparser,
    parse_sika_args,
    parse_path,
    write_out as _write_out,
    get_pool,
    file_timestamp
)


if __name__ == "__main__":
    parser = sika_argparser(description="Run this model", default_cfg_path=None)
    # add more args to the parser here if needed
    
    run_name, config_path, restore_from, outdir, args = parse_sika_args(parser)
    
    config = Config(config_path)
    logger = configure_logger("2mass_j1602")

    logger.parent.setLevel(logging.DEBUG)
    for handler in logger.parent.handlers:
        handler.setLevel(logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    def write_out(*args, level=logging.INFO):
        _write_out(*args, level=level, logger=logger)
    
    if restore_from is not None:
        write_out(f"Resuming model run from file {restore_from}")
        
    
    if outdir is None:
        outdir = join(parse_path(config["filestore"]), "out", "gl_orbit", f"{run_name}_{file_timestamp()}")
    
    os.makedirs(outdir, exist_ok=True)
    
    ecc = 0.2325
    period = (0.0332 * u.year).to_value("second")
    omega = 0.0432
    tau = 0.0710

    config["target"] = "Gl_Orbit"
    data_cfg = config["Gl_Orbit"]["data"]
    rv_path = data_cfg["rv_path"]
    rv_1_col, rv_2_col = data_cfg["rv_cols"]
    rv_a_err_col, rv_b_err_col = data_cfg["rv_err_cols"]
    t_col = data_cfg["time_col"]

    # MODEL_RUN = "gl_k_newpriors"
    # rv_path = join(config["filestore"],"out","gl229b",MODEL_RUN,"rv_corr.csv")
    assert exists(rv_path), f"Path {rv_path} does not exist"

    with open(join(outdir,"rv_info.txt"),"w") as f:
        f.write(f"RV data from {rv_path}\n")

    # shutil.copy(rv_path, outdir)

    rvs = pd.read_csv(rv_path)
    rv_a = rvs[rv_1_col] * 1e3
    rv_b = rvs[rv_2_col] * 1e3
    rv_a_err = rvs[rv_a_err_col] * 1e3
    rv_b_err = rvs[rv_b_err_col] * 1e3
    t = rvs[t_col] - min(rvs[t_col])
    t = np.array(t)
    t *= 86400  # convert to seconds
    
    BaBb_rv_params = BinaryRVParams(
        name="BaBb", 
        amplitude_1=Uniform(5e3,1e5),
        amplitude_2=Uniform(5e3,1e5),
        period=Normal(period,8640,bounds=(0,100*86400)),
        eccentricity=Parameter("eccentricity",NullPriorTransform(), frozen=True, values=ecc),
        # eccentricity=Normal(ecc,0.05,bounds=(0,1)),
        omega_planet=Uniform(0,2*np.pi),
        tau=Uniform(0,1),
        offset=Uniform(-1e3,1e3),
        drift=Uniform(0,0.001),
        # drift=Parameter("rv_drift",NullPriorTransform(), frozen=True, values=0)
    )
    
    aux = AuxiliaryParameterSet(jitter=Uniform(0,1e3))

    model = BinaryKeplerianRV(BaBb_rv_params, t-min(t), t-min(t))
    
    rv_a_ds = RV(t=t, rv=rv_a, rv_err=rv_a_err,parameters={})
    rv_b_ds = RV(t=t, rv=rv_b, rv_err=rv_b_err,parameters={})
    
    binary_rv_data = BinaryRV(rv1=rv_a_ds, rv2=rv_b_ds, parameters={})

    data = Dataset([binary_rv_data], dims=[])
    
    # pool = get_pool(config)
    
    sampler = BinaryRVSampler(run_name, outdir, data, [model], aux_params=aux)
    sampler.configure(config,logger)
    sampler.fit()