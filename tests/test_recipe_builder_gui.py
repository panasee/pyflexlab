import json

import pytest

from pyflexlab.recipe_builder_gui import (
    DEFAULT_MODULE_LIBRARY,
    GuiRecipeSpec,
    ModuleInstance,
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
