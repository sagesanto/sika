# Keck Planet Imager and Characterizer (KPIC)

This section details the directory structure and config keys that `sika` KPIC functionality expects.

## Configuration
These entries should appear in the `sika` config file.
```
[kpic]
data_dir = kpic science data root directory
calib_dir = kpic calibration root directory (for responses)
```

For each companion:

```
[{target_name}]
    primary = '{primary_name}'
    [{target_name}.data]
        orders = [{order},{order},...]
        fibers = ['{fiber}', '{fiber}', ...]
        nights = ['{night}','{night}', ...]
        [{target_name}.data.{night}]
            response_file = '{response file}'
            response_star = '{response star}'
            [{target_name}.data.{night}.exposures]
                {fiber} = ['{exposure}', '{exposure}', ...]
                {fiber} = ['{exposure}', '{exposure}', ...]
                ...
```

Where:

- `{target_name}` is the name of a target (ex. `Gl229B`)

- `{primary_name}` (optional) if this target is a companion, this is the name of its host star. a separate config entry of the same format for the host star should appear in the config file, where its `{target_name}` is identical to this `{primary_name}`. 

- `{response file}` is a relative path to the response file (see file structure below). This path should be with respect to `{calib_dir}`/`{night}`, where `{calib_dir}` is defined in the [kpic] config section above. This does not need to be provided if `{response star}` is provided.

- `{response star}` is the name of the response star. If this is provided, it is assumed that the respective response can be found at `{science_dir}`/`{response star}`/`{night}`/'`{night}`_spec_response_`{fiber}`.fits', where `{science_dir}` is defined in the [kpic] config section above and `{fiber}` is the name of a fiber, ex. `{sf2}`. This does not need to be provided if `{response star}` is provided.


- `{night}` is the name of an observing night in the format `YYYYMMDD` (ex. `20250616`)

- `{order}` is an integer index of a spectral order that should be included in the spectrum

- `{exposure}` is a number (ex. `0164`) that corresponds to a frame that should be included in the processing, where the same identifier appears in the exposure fits filenames (see below)

- `{fiber}` is the name of a fiber that should be included, ex. `SF1`


## File Structure
This section lays out the required on-disk file structure for data and calibrations.

### Data

```
<data_dir>
├── {target_name}
│   ├── {night}
│   │   ├──{spectral response file}.fits
│   │   └── fluxes
│   │       └── {fits files}
│   └── {night}
│       └── fluxes
│           └── {fits files}
└── {target_name}
    └── {night}
        ├──{spectral response file}.fits
        └── fluxes
            └── {fits files}
    ...
```

Where:

- `{data dir}` matches the data directory specified in the `[kpic]` config section (see above)

- `{target_name}` is the name of a target (ex. `Gl229B`)

- `{night}` is the name of an observing night in the format `YYYYMMDD` (ex. `20250616`)

- `{fits files}` are a list of fits files that contain `_{exposure}_`, where `{exposure}` is as defined above, and end in `.fits`

- `{spectral response file}` is a KPIC response file containing a kpicdrp.drp_data.Spectrum response for each fiber

### Calibrations
`sika` will attempt to use the `kpicdrp` calibration database to load calibrations. The following directory structure provides a necessary fallback that allows the module to work without the calibration database.

```
{calib_dir}
└── {night}
    ├── {wavecal file}.fits
    └── {trace file}.fits
```
Where:

 - `{calib dir}` matches the calibration directory specified in the `[kpic]` config section (see above)

 - `{night}` is the name of an observing night in the format `YYYYMMDD` (ex. `20250616`)

 - `{wavecal file}` is a kpicdrp-readable wavecal file **whose name ends in `_wvs.fits`** (ex. `20250616_HIP62944_psg_wvs.fits`)

 - `{trace file}` is a kpicdrp-readable trace file **whose name ends in `_trace.fits`** (ex. `nspec250616_0143_trace.fits`)