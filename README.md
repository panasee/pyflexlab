# pyflexlab 🔬

<div align="center">

[![Version](https://img.shields.io/badge/version-2.0-blue.svg)](pyproject.toml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![PyMeasure](https://img.shields.io/badge/PyMeasure-0.14.0%2B-orange.svg)](https://github.com/pymeasure/pymeasure)
[![QCoDeS](https://img.shields.io/badge/QCoDeS-0.47.0%2B-green.svg)](https://github.com/microsoft/Qcodes)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Flexible laboratory measurement workflows for instrument control, recording, and live visualization**

</div>

PyFlexLab is a laboratory measurement package for flexible instrument control,
data recording, live plotting, and post-processing. It wraps PyMeasure/QCoDeS
drivers behind local workflow utilities, while keeping enough low-level control
for custom condensed-matter and cryogenic measurement scripts.

> **Why PyFlexLab instead of plain QCoDeS/PyMeasure?** PyFlexLab keeps the
> direct-control feel of low-level instrument work, while adding local
> conventions for file organization, measurement recipes, live plotting, and
> workflow templates.

---

## 📋 Table Of Contents

- [✨ Key Features](#-key-features)
- [🧭 What's New In 2.0](#-whats-new-in-20)
- [🚀 Installation](#-installation)
- [⚙️ Environment Setup](#️-environment-setup)
- [📖 Measurement Workflows](#-measurement-workflows)
  - [Recipe Runner](#recipe-runner)
  - [Modular Builders](#modular-builders)
  - [Direct Recipes](#direct-recipes)
- [🖥️ GUIs](#️-guis)
- [🗂️ Data And Templates](#️-data-and-templates)
- [🔌 Supported Instruments](#-supported-instruments)
- [⚠️ Known Notes](#️-known-notes)
- [🧪 Development Checks](#-development-checks)

## ✨ Key Features

- 🔧 **Recipe-based measurement flow**: describe measurement structure with
  `MeasurementRecipe`, then execute through `MeasureFlow.run_recipe()`.
- 🧩 **Modular builders**: compose source, sense, external-control, and plot
  fragments through `RecipeBuilder` and `MeasureModules`.
- 📊 **Live visualization**: use Jupyter inline plotting for short runs or Dash
  plotting for longer measurements.
- 💾 **Structured data storage**: organize local templates and output records
  through `PYLAB_DB_LOCAL` and `PYLAB_DB_OUT`.
- 🖥️ **GUI tooling**: launch the main measurement GUI, coordinate-transform GUI,
  or experimental recipe-spec builder.
- 🔌 **Driver bridge**: combine PyMeasure/QCoDeS drivers with lab-specific
  wrappers under a unified local workflow.

## 🧭 What's New In 2.0

Version 2.0 marks the first recipe-based measurement workflow in this project.
Legacy `MeasureFlow.measure_*_legacy` methods are still available, but new
workflows should prefer:

- `MeasurementRecipe` and `PlotRecipe` from `pyflexlab.measure_flow`
- `MeasureFlow.run_recipe()` as the common execution boundary
- `RecipeBuilder`, `RecipeOptions`, `MeasureModules`, and `PlotModules` from
  `pyflexlab.recipe_builders`
- `PlotModules.mapped_plot()` with declarative `PlotSeries` column mappings
- `pyflexlab.recipe_builder_gui` for visual recipe-spec editing

The current template also supports external `T`, `B`, and `Theta` controls in
fixed, sweep, and vary forms.

## 🚀 Installation

Install from a local checkout:

```bash
python -m pip install .
```

Install with GUI dependencies:

```bash
python -m pip install ".[gui]"
```

Install from PyPI when the package release is available:

```bash
python -m pip install pyflexlab
```

## ⚙️ Environment Setup

PyFlexLab uses two database roots:

- `PYLAB_DB_LOCAL`: local templates and relatively stable configuration files
- `PYLAB_DB_OUT`: measurement output, project folders, records, and generated data

The package also accepts suffixed variants such as `PYLAB_DB_OUT_USER`; when
multiple matching variables exist, the first one in alphabetical order is used.
For fully explicit paths, set both `PYLAB_LOCAL_SPECIFIC` and
`PYLAB_OUT_SPECIFIC` before importing `pyflexlab`.

Initialize template files into the local database:

```python
import pyflexlab

pyflexlab.initialize_with_templates()
```

This seeds files such as `measure_types.json`, `assist_measure.ipynb`, and
`assist_post.ipynb`. Existing local files are not overwritten.

## 📖 Measurement Workflows

### Recipe Runner

The 2.0 workflow describes a measurement as a `MeasurementRecipe` and executes
it with `MeasureFlow.run_recipe()`. The runner owns the common measurement
mechanics:

- call `MeasureManager.get_measure_dict()`
- iterate measurement generators
- write records with `record_update()`
- update optional live plots
- run hooks
- shut down requested outputs

Supported hooks are:

- `after_prepare(mea_dict)`
- `on_measure(record_tuple)`
- `on_record(record_tuple)`
- `plot.update(plotobj, record_tuple)`
- `shutdown` targets

`record_update()` is part of the fixed runner pipeline, not a user hook.

### Modular Builders

`pyflexlab.recipe_builders` provides small module fragments that assemble into a
`MeasurementRecipe` while preserving the underlying `get_measure_dict()` order:
sources first, then senses, then external controls.

```python
from pyflexlab import MeasureFlow
from pyflexlab.recipe_builders import (
    MeasureModules,
    PlotModules,
    PlotSeries,
    RecipeBuilder,
    RecipeOptions,
)

flow = MeasureFlow("demo")
flow.load_fakes(2)
source_meter, sense_meter = flow.instrs["fakes"]

plot = PlotModules.mapped_plot(
    init_args=(1, 1, 1),
    titles=[["B I"]],
    series=[
        PlotSeries(
            row=0,
            col=0,
            line=0,
            x_col=3,
            y_col=1,
            x_label="B",
            y_label="I",
        )
    ],
    use_dash=True,
)

recipe = (
    RecipeBuilder(
        options=RecipeOptions(
            if_combine_gen=True,
            special_name="demo",
            vary_loop=True,
            source_wait=0,
            wait_before_vary=5,
        )
    )
    .add(
        MeasureModules.fixed_current_source(
            1e-6,
            high=0,
            low=0,
            meter=source_meter,
            compliance=1,
        )
    )
    .add(MeasureModules.voltage_sense(high=0, low=0, meter=sense_meter, ac_dc="dc"))
    .add(MeasureModules.vary_magnetic_field(start=-1, stop=1))
    .add(MeasureModules.fixed_temperature(300))
    .build(step_time=0, plot=plot, shutdown=(source_meter,))
)

result = flow.run_recipe(recipe)
print(result["file_path"])
```

Available builder module groups:

- **Sources**: fixed/sweep current source, fixed/sweep voltage source
- **Senses**: current sense, voltage sense
- **External controls**: magnetic field, temperature, and angle as fixed/sweep/vary
- **Plots**: declarative `PlotSeries` mappings through `PlotModules.mapped_plot()`

For complete runnable examples, see `pyflexlab/templates/assist_measure.ipynb`.

### Direct Recipes

For workflows that need tighter control than the builder provides, construct
`MeasurementRecipe` and `PlotRecipe` directly from `pyflexlab.measure_flow`.
This is the stable boundary when porting one legacy `measure_*` method at a
time.

## 🖥️ GUIs

Installed console scripts:

```bash
gui_measure
gui_coor_trans
gui_recipe_builder
```

GUI surfaces:

- 🔬 `gui_measure`: main PyFlexLab measurement studio
- 🧭 `gui_coor_trans`: coordinate transformation helper for sample navigation
- 🧩 `gui_recipe_builder`: experimental recipe-spec builder with
  module drag/drop, parameter editing, ordering inside source/sense/external/plot
  boxes, JSON preview, and optional Dash preview

Launch the recipe builder GUI from Python:

```python
from pyflexlab.recipe_builder_gui import main

main()
```

The recipe builder GUI currently edits a structured spec. Translation from that
spec into real instrument-bound `MeasurementRecipe` objects remains an explicit
step.

## 🗂️ Data And Templates

`measure_types.json` defines naming and sweep/vary metadata for measurement
types. The current template supports external `T`, `B`, and `Theta` controls in
fixed, sweep, and vary forms.

`assist_measure.ipynb` is the primary interactive measurement template. It now
contains both direct recipe examples and a modular recipe-builder flow.

`assist_post.ipynb` is the post-processing template.

New project creation copies notebooks from the configured local database, so run
`initialize_with_templates()` after installing or updating the package templates.

## 🔌 Supported Instruments

Supported or wrapped instruments include:

- **Meters and source meters**: Keithley 2182A, 2400, 2401, 2450, 6221, 6430,
  6500; Keysight B2902B; SR830; SR860
- **Temperature controllers**: Oxford ITC503, Oxford Mercury ITC, Lake Shore 336
- **Magnet controllers**: Oxford IPS
- **Other hardware**: probe rotator and local lab-specific devices

Some drivers are adapted from PyMeasure/QCoDeS and some are lab-specific. Add
new wrappers by following the abstractions under `pyflexlab.equip_wrapper`.

## ⚠️ Known Notes

- For long measurements, prefer Dash plotting over inline Jupyter plotting.
- `measure_flow_old.py` remains as the legacy method collection.
- New public workflows should be thin wrappers that build recipes and delegate
  execution to `run_recipe()`.
- The recipe builder GUI is an editing surface, not yet a full instrument
  execution front end.

## 🧪 Development Checks

Useful focused checks during recipe/GUI work:

```bash
python -m pytest tests/test_recipe_builder_gui.py -q
python -m py_compile pyflexlab/recipe_builder_gui.py pyflexlab/recipe_builders.py
```

Use the Python executable from the project environment when working on the local
Windows checkout.

---

<div align="center">
  <sub>Built for flexible lab automation and scientific measurement workflows.</sub>
</div>
