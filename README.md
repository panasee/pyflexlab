# pylab_dk

**pylab_dk** is an integrated package based on [PyMeasure](https://github.com/pymeasure/pymeasure) and [QCoDeS](https://github.com/microsoft/Qcodes),
designed for collecting, processing, and plotting experimental data.
(**why not use QCoDeS directly:** allow for more flexible measurements and plotting at lower level)

## Table of Contents

- [Installation](#installation)
- [Key Features](#key-features)
- [Usage](#usage)
    - [Set Environmental Variables](#set-environmental-variables)
    - [Create Own measure_types.json](#create-own-measure_typesjson)
    - [Create Own Jupyter Notebook](#create-own-jupyter-notebook)
    - [Provided GUIs](#provided-guis)
    - [Provided Template](#provided-template)
- [Known Issues](#known-issues)
- [Dependencies](#dependencies)

## Key Features
- (most importantly) Lower-level interface and modularized components, enabling flexible composition of complex functionalities.
- Unified interface for instruments control
- Structured data storage with automatic file organization
- Real-time visualization and data recording (Jupyter / Dash web interface)
- Attached with data analysis and processing methods

## Installation

Ensure you have Python 3.11 or higher installed. Virtual environment is recommended (but won't be introduced here). You can install the required packages locally using pip:
```bash
cd ${PACKAGE_DIR}
python -m pip install .  # or `pip install .[gui]` for using GUI
```

## Usage
### set environmental variables
- PYLAB_DB_LOCAL: the path to the local database, storing rarely changing data like measure_type.json
- PYLAB_DB_OUT: the path to the out database, storing the main experimental data and records

(set them via `os.environ` or directly in the system setting, the suffixes of the env vars do not matter, like `PYLAB_DB_OUT` and `PYLAB_DB_OUT_User` are the same. The package will automatically choose the first one found)
### Create own measure_types.json
 A file named "measure_types.json" is used for automatically organizing data files. This highly depends on personal preferences and research needs, so a template is provided here. 
 The finished file should be placed in the local database directory specified by [PYLAB_DB_LOCAL](#set-environmental-variables).

**Naming rules:** use `{variable}` to represent variables that need to be replaced when naming.
```json
{        
  "V": {
        "sense": "V{note}-{vhigh}-{vlow}",
        "source": {
            "fixed":{
                "dc": "Vfix{fixv}V-{vhigh}-{vlow}",
                "ac": "Vfix{fixv}V-freq{freq}Hz-{vhigh}-{vlow}"
            },
            "sweep": {
                "dc": "Vmax{maxv}V-step{stepv}V-{vhigh}-{vlow}-swpmode{mode}",
                "ac": "Vmax{maxv}V-step{stepv}V-freq{freq}Hz-{vhigh}-{vlow}"
            }
        }
    },
    "I": {
        "sense": "I{note}-{iin}-{iout}",
        "source": {
            "fixed": {
                "dc": "Ifix{fixi}A-{iin}-{iout}",
                "ac": "Ifix{fixi}A-freq{freq}Hz-{iin}-{iout}"
            },
            "sweep": {
                "dc": "Imax{maxi}A-step{stepi}A-{iin}-{iout}-swpmode{mode}",
                "ac": "Imax{maxi}A-step{stepi}A-freq{freq}Hz-{iin}-{iout}"
            }
        }
    },
  "T": {
        "fixed": "Temp{fixT}K",
        "sweep": "Temp{Tstart}-{Tstop}K-step{stepT}K-swpmode{mode}",
        "vary": "Temp{Tstart}-{Tstop}K"
    }
}
```

### Create own jupyter notebook (not necessary)
Jupyter Notebook is useful for data analysis and plotting.
When creating a new project, the package will copy the templates named `assist_measure.ipynb` and `assist_post.ipynb` to the project directory for convenience. But if there is no template, the package will not throw error.

As for real-time data plotting, the package will use `dash` to create a web app for plotting when not using jupyter notebook.

### Provided GUIs
- "gui-coor-trans": a GUI for coordinate transformation used to locate objects using two reference points on a flat surface
- "gui-pan-color": a color palette for choosing colors 

### Provided template
See the provided [template](template.md) for detailed template which contains most of the common measurement types.

## Known issues
- The driver of the rotator is not working properly
- Currently no keyboard interruption actions implemented, if the measurement is interrupted, the meters would be left in the last state.(data is saved in real-time, interruption won't affect data)
- The `dash` app in Chrome would crash from time to time. This won't affect anything, just refresh the page.
- In jupyter notebook, the `plotly` package takes a lot of memory especially when there are a huge amount of data points. In some extreme cases (~$10^5$ points), the python kernel would crash and interrupt the measurement.

## dependencies
- python >= 3.11 (earlier version is not tested)
- Required packages:
  - numpy
  - pandas
  - matplotlib
  - plotly >= 5.24.1
  - kaleido == 0.1.0.post1
  - pyvisa
  - pyvisa-py
  - pymeasure >= 0.14.0
  - qcodes >= 0.47.0
  - jupyter
  - dash
- Optional packages:
  - PyQt6
