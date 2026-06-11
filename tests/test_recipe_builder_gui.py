import json

import pytest

from pyflexlab.recipe_builder_gui import (
    DEFAULT_MODULE_LIBRARY,
    GuiRecipeSpec,
    ModuleInstance,
    _parse_parameters_json,
    _spec_has_live_plot,
)


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
