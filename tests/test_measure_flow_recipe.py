from pyflexlab.measure_flow import MeasureFlow, MeasurementRecipe
from pyflexlab.measure_flow_old import MeasureFlow as LegacyMeasureFlow


def test_new_measure_flow_preserves_legacy_measure_methods():
    assert issubclass(MeasureFlow, LegacyMeasureFlow)
    assert hasattr(MeasureFlow, "measure_Vswp_I_vicurve")


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
        get_measure_kwargs={"special_name": "recipe-test"},
    )

    result = flow.run_recipe(recipe)

    assert result["file_path"] == "fake.csv"
    assert flow.rows == [
        ("fake.csv", 2, (0.0, 1.0), False),
        ("fake.csv", 2, (0.1, 1.1), False),
    ]
