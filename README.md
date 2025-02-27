# pyflexlab 🔬

<div align="center">

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyMeasure](https://img.shields.io/badge/PyMeasure-0.14.0+-orange.svg)](https://github.com/pymeasure/pymeasure)
[![QCoDeS](https://img.shields.io/badge/QCoDeS-0.47.0+-green.svg)](https://github.com/microsoft/Qcodes)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An integrated package for scientific data collection, processing, and visualization**

</div>

**pyflexlab** is an integrated package based on [PyMeasure](https://github.com/pymeasure/pymeasure) and [QCoDeS](https://github.com/microsoft/Qcodes),
designed for collecting, processing, and plotting experimental data.

> **Why pyflexlab instead of QCoDeS/PyMeasure?**  While QCoDeS/PyMeasure provides excellent out-of-box experience for experiments, pyflexlab offers a more flexible, lower-level interface that enables direct control over instruments, as if sitting right beside them. This approach allows researchers to implement highly customized experimental workflows. *More to mention, the drivers of QCoDeS or PyMeasure alone are not complete. It's often the case one needs to assemble different versions of drivers.*

> **Brief Summary of What it does:** Utilize the basic interface of PyMeasure & QCoDeS to manipulate instruments and encapsulate similar instruments with unified interface, thus providing basic modules for measurement workflow.

## 📋 Table of Contents

- [🚀 Installation](#-installation)
- [✨ Key Features](#-key-features)
- [📖 Usage](#-usage)
    - [Set Environmental Variables](#set-environmental-variables)
    - [Configure Local Setting Files](#configure-local-setting-files)
    - [Start Measurement](#start-measurement)
    - [Provided GUIs](#provided-guis)
- [⚠️ Known Issues](#️-known-issues)
- [📦 Dependencies](#-dependencies)

## ✨ Key Features
- 🔧 **Flexible Architecture**: Lower-level interface and modularized components, enabling flexible composition of complex functionalities
- 🎛️ **Unified Control**: Seamless interface for instruments control
- 💾 **Automatic File Organization**: Structured data storage with automatic file organization
- 📊 **Live Visualization**: Real-time visualization and data recording (Jupyter / Dash web interface)
- 📈 **Data Analysis**: Comprehensive tools for data analysis and processing

## 🚀 Installation

Ensure you have Python 3.11 or higher installed. Virtual environment is recommended. You can install it through pypi or local package downloaded here (add [gui] option for using GUI):

```bash
# local package installation
cd ${PACKAGE_DIR}  # replace with folder path
python -m pip install . 
# pypi installation
python -m pip install pyflexlab 
```

## 📖 Usage

### Set Environmental Variables

- `PYLAB_DB_LOCAL`: the path to the local database, storing rarely changing data like `measure_type.json`
- `PYLAB_DB_OUT`: the path to the out database, storing the main experimental data and records

(set them via `os.environ` or directly in the system setting, the suffixes of the env vars do not matter, like `PYLAB_DB_OUT` and `PYLAB_DB_OUT_User` are the same. The package will automatically choose the first one found in alphabetic order)

### Configure Local Setting Files

`measure_types.json` is used for automatically naming data files. This highly depends on personal preferences and research needs. A template is provided. 
`assist_measure.ipynb` and `assist_post.ipynb` are jupyter notebook files used for convenience of measurements. The program will automatically retrieve them from local database path and paste them to the measurement directory.

These files should be placed in the local database directory specified by [PYLAB_DB_LOCAL](#set-environmental-variables).

Also, templates for them are provided with the package, run follow command to initialize the local database folder:
```python
import pyflexlab
pyflexlab.initialize_with_templates()
```

### Start Measurement

Detailed examples of a few typical measurements have been demonstrated in `assist_measure.ipynb`. But just as the reason why this package exists, the measurement workflow can be easily established via basic python knowledge and understanding of instruments.

### Provided GUIs

- 🔄 **gui-coor-trans**: A GUI for coordinate transformation used to locate objects using two reference points on a flat surface (linear transform solver)
- 🎨 **gui-pan-color**: A color palette for choosing colors 

## ⚠️ Known Issues

- The memory management of plotly drawing remains a problem for long-time measurement. (be careful when number of points exceeds 50k)
- The driver of the rotator is not working properly due to weird interaction between C++ dll and python
- Currently no keyboard interruption actions implemented, if the measurement is interrupted, the meters would be left in the last state (data is saved in real-time, interruption won't affect data)
- The `dash` app in Chrome would crash from time to time. (won't affect anything, just refresh the page)

## 📦 Dependencies

- Python >= 3.11 (earlier version is not tested)
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

---

<div align="center">
  <sub>Built with ❤️ for scientific research</sub>
</div>
