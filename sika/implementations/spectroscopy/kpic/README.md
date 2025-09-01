# KPIC

## Data Format
This section details the directory structure and config keys that KPIC sika functionality expects. `<keywords>` are the names of config keys.

### Data config
```
[kpic]
data_dir = kpic science data root directory
calib_dir = kpic calibration root directory (for responses)
```

For each target:
```
[{target_name}]
    primary = {primary_name}
    [{target_name}.data]
        [{target_name}.data.{night}]
            exposures = ['{exposure}', '{exposure}', ...]
            fibers = ['{fiber}', '{fiber}', ...]
```

Where:
- `{target_name}` is the name of a target (ex. `Gl229B`)
- `{primary_name}` (optional) is the name of the star, with a config of the same format. 
- `{night}` is the name of an observing night in the format `YYYYMMDD` (ex. `20250616`)
- `{exposure}` is a number (ex. `0164`) that corresponds to a frame that should be included in the processing, where the same identifier appears in the exposure fits filenames (see below)
- `{fiber}` is the name of a fiber that should be included, ex. `SF1`

### Data directory:

```
<data_dir>
├── {target_name}
│   ├── {night}
│   │   └── fluxes
│   │       └── {fits files}
│   └── {night}
│       └── fluxes
│           └── {fits files}
└── {target_name}
    └── {night}
        └── fluxes
            └── {fits files}
```

Where:
- `{fits files}` are a list of fits files that contain `_{exposure}_`, where `{exposure}` is as defined above, and end in `.fits`

### Calibration directory:

```
<calib_dir>
└── {night}
   └── {response file}
```
Where:
 - `{response file}` is a file containing the response **expand on name and data format**