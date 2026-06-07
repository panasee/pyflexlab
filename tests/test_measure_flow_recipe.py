import uuid
import inspect
from pathlib import Path

from pyflexlab.measure_flow import MeasureFlow, MeasurementRecipe
from pyflexlab.measure_flow_old import MeasureFlow as LegacyMeasureFlow


def test_legacy_measure_methods_are_suffixed():
    assert issubclass(MeasureFlow, LegacyMeasureFlow)
    assert hasattr(MeasureFlow, "measure_Vswp_I_vicurve_legacy")
    assert not hasattr(MeasureFlow, "measure_Vswp_I_vicurve")

    legacy_measure_names = [
        name
        for name, _ in inspect.getmembers(LegacyMeasureFlow, predicate=inspect.isfunction)
        if name.startswith("measure_") or name.startswith("b2_measure_")
    ]
    assert legacy_measure_names
    assert all(name.endswith("_legacy") for name in legacy_measure_names)


def test_run_recipe_records_rows_from_measure_dict():
    class FakeMeasureFlow(MeasureFlow):
        def __init__(self):
            self.rows = []

        def get_measure_dict(self, measure_mods, *args, **kwargs):
            assert measure_mods == ("V_sense_dc",)
            assert args == (1, 2)
            assert kwargs["wrapper_lst"] == ["meter"]
            assert kwargs["compliance_lst"] == [None]
            assert kwargs["special_name"] == "recipe-test"
            return {
                "gen_lst": iter([(0.0, 1.0), (0.1, 1.1)]),
                "file_path": "fake.csv",
                "plot_record_path": "fake_plot.csv",
                "record_num": 2,
            }

        def record_update(self, file_path, record_num, record_tuple, *, force_write=False):
            self.rows.append((file_path, record_num, record_tuple, force_write))

    flow = FakeMeasureFlow()
    recipe = MeasurementRecipe(
        measure_mods=("V_sense_dc",),
        args=(1, 2),
        wrapper_lst=["meter"],
        compliance_lst=[None],
        measure_kwargs={"special_name": "recipe-test"},
    )

    result = flow.run_recipe(recipe)

    assert result["file_path"] == "fake.csv"
    assert flow.rows == [
        ("fake.csv", 2, (0.0, 1.0), False),
        ("fake.csv", 2, (0.1, 1.1), False),
    ]


def test_run_recipe_records_complete_fake_source_sense_external_recipe():
    flow = MeasureFlow("test")
    flow.load_fakes(2)

    source_meter, sense_meter = flow.instrs["fakes"]
    appendix = f"-fake-recipe-{uuid.uuid4().hex[:8]}"
    recipe = MeasurementRecipe(
        measure_mods=(
            "V_source_sweep_dc",
            "I_sense_dc",
            "B_fixed",
            "T_fixed",
        ),
        args=(0.1, 0.1, 1, 0, "0-max-0", "", 1, 0, 0, 300),
        wrapper_lst=[source_meter, sense_meter],
        compliance_lst=[1e-3],
        measure_kwargs={
            "with_timer": False,
            "source_wait": 0,
            "special_name": "fake-full-run",
            "appendix_str": appendix,
        },
    )

    try:
        result = flow.run_recipe(recipe)
    finally:
        flow.record_finalize()

    csv_lines = Path(result["file_path"]).read_text(encoding="utf-8").splitlines()
    assert result["record_num"] == 4
    assert csv_lines[0] == "V_source,I,B,T"
    assert len(csv_lines) > 1
    assert flow.proj_name == "test"
    assert source_meter.info_dict["output_status"] is False
