import json

import pytest

from pyflexlab.recipe_builder_gui import (
    DEFAULT_MODULE_LIBRARY,
    GuiRecipeSpec,
    ModuleInstance,
    _parse_parameters_json,
    _spec_has_live_plot,
)
from pyflexlab.recipe_builders import MeasureModules


def test_default_module_library_has_four_gui_boxes():
    categories = {module.category for module in DEFAULT_MODULE_LIBRARY}

    assert categories == {"source", "sense", "external", "plot"}


def test_recipe_spec_rejects_module_in_wrong_box():
    source_module = next(
        module for module in DEFAULT_MODULE_LIBRARY if module.category == "source"
    )

    with pytest.raises(ValueError, match="source module cannot be added to sense"):
        GuiRecipeSpec(senses=[ModuleInstance.from_definition(source_module)])


def test_recipe_spec_exports_structured_json():
    source_module = next(
        module for module in DEFAULT_MODULE_LIBRARY if module.category == "source"
    )
    external_module = next(
        module for module in DEFAULT_MODULE_LIBRARY if module.category == "external"
    )

    recipe = GuiRecipeSpec(
        sources=[ModuleInstance.from_definition(source_module)],
        externals=[ModuleInstance.from_definition(external_module)],
    )

    exported = json.loads(recipe.to_json())

    assert exported["sources"][0]["module_id"] == source_module.module_id
    assert exported["externals"][0]["category"] == "external"
    assert exported["senses"] == []


def test_only_live_xy_plot_enables_dash_preview():
    live_plot = next(
        module for module in DEFAULT_MODULE_LIBRARY if module.module_id == "plot.rh_loop"
    )
    record_only = next(
        module
        for module in DEFAULT_MODULE_LIBRARY
        if module.module_id == "plot.record_only"
    )

    assert _spec_has_live_plot(
        GuiRecipeSpec(plots=[ModuleInstance.from_definition(live_plot)])
    )
    assert not _spec_has_live_plot(
        GuiRecipeSpec(plots=[ModuleInstance.from_definition(record_only)])
    )


def test_gui_library_uses_recipe_builder_module_ids():
    module_ids = {module.module_id for module in DEFAULT_MODULE_LIBRARY}

    assert {
        "source.fixed_current",
        "source.fixed_voltage",
        "sense.voltage",
        "sense.current",
        "external.vary_magnetic_field",
        "external.fixed_temperature",
        "plot.rh_loop",
        "plot.vi_curve",
    }.issubset(module_ids)


def test_gui_library_matches_recipe_builder_measure_module_ids():
    meter = object()
    builder_modules = (
        MeasureModules.fixed_current_source(
            0, high=0, low=0, meter=meter, compliance=1
        ),
        MeasureModules.fixed_voltage_source(
            0, high=0, low=0, meter=meter, compliance=1
        ),
        MeasureModules.sweep_current_source(
            0, 0, high=0, low=0, sweepmode="0-max-0", meter=meter, compliance=1
        ),
        MeasureModules.sweep_voltage_source(
            0, 0, high=0, low=0, sweepmode="0-max-0", meter=meter, compliance=1
        ),
        MeasureModules.voltage_sense(high=0, low=0, meter=meter, ac_dc="dc"),
        MeasureModules.current_sense(high=0, low=0, meter=meter, ac_dc="dc"),
        MeasureModules.fixed_magnetic_field(0),
        MeasureModules.vary_magnetic_field(start=0, stop=0),
        MeasureModules.sweep_magnetic_field(
            start=0, stop=0, step=0, sweepmode="0-max-0"
        ),
        MeasureModules.fixed_temperature(0),
        MeasureModules.vary_temperature(start=0, stop=0),
        MeasureModules.sweep_temperature(
            start=0, stop=0, step=0, sweepmode="0-max-0"
        ),
        MeasureModules.fixed_angle(0),
        MeasureModules.vary_angle(start=0, stop=0),
        MeasureModules.sweep_angle(start=0, stop=0, step=0, sweepmode="0-max-0"),
    )
    builder_ids = {module.module_id for module in builder_modules}
    gui_measure_ids = {
        module.module_id
        for module in DEFAULT_MODULE_LIBRARY
        if module.category != "plot"
    }

    assert gui_measure_ids == builder_ids


def test_gui_library_default_parameters_match_builder_factories():
    default_keys_by_id = {
        module.module_id: set(module.default_parameters)
        for module in DEFAULT_MODULE_LIBRARY
    }

    assert default_keys_by_id["source.fixed_current"] == {
        "value",
        "high",
        "low",
        "meter",
        "compliance",
        "freq",
    }
    assert default_keys_by_id["source.fixed_voltage"] == {
        "value",
        "high",
        "low",
        "meter",
        "compliance",
        "freq",
    }
    assert default_keys_by_id["source.sweep_current"] == {
        "max_value",
        "step_value",
        "high",
        "low",
        "sweepmode",
        "meter",
        "compliance",
        "freq",
    }
    assert default_keys_by_id["source.sweep_voltage"] == {
        "max_value",
        "step_value",
        "high",
        "low",
        "sweepmode",
        "meter",
        "compliance",
        "freq",
    }
    assert default_keys_by_id["sense.voltage"] == {
        "high",
        "low",
        "comment",
        "meter",
        "ac_dc",
    }
    assert default_keys_by_id["sense.current"] == {
        "high",
        "low",
        "comment",
        "meter",
        "ac_dc",
    }
    assert default_keys_by_id["external.fixed_angle"] == {"value"}
    assert default_keys_by_id["external.vary_angle"] == {"start", "stop"}
    assert default_keys_by_id["external.sweep_angle"] == {
        "start",
        "stop",
        "step",
        "sweepmode",
    }


def test_recipe_spec_can_reorder_modules_within_a_box():
    fixed_voltage = next(
        module
        for module in DEFAULT_MODULE_LIBRARY
        if module.module_id == "source.fixed_voltage"
    )
    fixed_current = next(
        module
        for module in DEFAULT_MODULE_LIBRARY
        if module.module_id == "source.fixed_current"
    )
    spec = GuiRecipeSpec(
        sources=[
            ModuleInstance.from_definition(fixed_voltage),
            ModuleInstance.from_definition(fixed_current),
        ]
    )

    spec.move_module("source", 1, 0)

    assert [module.module_id for module in spec.sources] == [
        "source.fixed_current",
        "source.fixed_voltage",
    ]


def test_recipe_spec_updates_module_parameters_by_category_and_index():
    source_module = next(
        module
        for module in DEFAULT_MODULE_LIBRARY
        if module.module_id == "source.fixed_current"
    )
    spec = GuiRecipeSpec(sources=[ModuleInstance.from_definition(source_module)])

    spec.update_module_parameters("source", 0, {"value": 1.5, "high": 1, "low": 0})

    assert spec.sources[0].parameters["value"] == 1.5
    assert spec.sources[0].parameters["high"] == 1
    assert source_module.default_parameters["value"] == 0


def test_parameter_json_parser_requires_object():
    assert _parse_parameters_json('{"value": 1, "high": 0}') == {
        "value": 1,
        "high": 0,
    }

    with pytest.raises(ValueError, match="parameters JSON must be an object"):
        _parse_parameters_json("[1, 2, 3]")
