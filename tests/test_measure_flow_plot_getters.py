import numpy as np
import pytest
from pyomnix.data_process import DataManipulator

from pyflexlab.measure_flow import MeasurementRecipe, MeasureFlow, PlotRecipe
from pyflexlab.recipe_builders import PlotModules, PlotSeries, RecipeBuilder, assemble_recipe


class FakePlot:
    def stop_saving(self) -> None:
        pass


class RecordingFlow(MeasureFlow):
    def __init__(self, record_num: int = 2) -> None:
        self.recorded = []
        self.record_num = record_num

    def prepare_recipe(self, recipe: MeasurementRecipe) -> dict:
        return {
            "gen_lst": iter(((1, 2), (3, 4))),
            "file_path": "unused.csv",
            "plot_record_path": "unused.png",
            "record_num": self.record_num,
            "vary_mod": [],
        }

    def record_update(self, file_path, record_num, record_tuple, **kwargs) -> None:
        assert len(record_tuple) == record_num
        self.recorded.append(record_tuple)

    def _init_recipe_plot(self, plotrec: PlotRecipe | None, mea_dict: dict):
        return FakePlot()


class PreparingFlow(MeasureFlow):
    def __init__(self) -> None:
        self.measure_kwargs = {}

    def get_measure_dict(self, measure_mods, *args, **kwargs):
        self.measure_kwargs = kwargs
        return {
            "gen_lst": iter(()),
            "file_path": "unused.csv",
            "plot_record_path": "unused.png",
            "record_num": 0,
            "vary_mod": [],
        }


def test_plot_getters_extend_plot_records_without_recording():
    flow = RecordingFlow()
    getter_values = iter((99, 100))
    seen_by_record = []
    seen_by_plot = []

    def on_record(records):
        seen_by_record.append(records)

    def update(plotobj, records):
        seen_by_plot.append(records)

    recipe = MeasurementRecipe(
        measure_mods=("I_source_sweep_dc", "V_sense_dc"),
        plot=PlotRecipe(
            update=update,
            extra_getters=(lambda: next(getter_values),),
        ),
        on_record=on_record,
    )

    flow.run_recipe(recipe)

    assert flow.recorded == [(1, 2), (3, 4)]
    assert seen_by_record == [(1, 2), (3, 4)]
    assert seen_by_plot == [(1, 2, 99), (3, 4, 100)]


def test_prepare_recipe_passes_record_getter_columns():
    flow = PreparingFlow()

    flow.prepare_recipe(
        MeasurementRecipe(
            measure_mods=("I_source_sweep_dc", "V_sense_dc"),
            extra_record_columns=("aux",),
            extra_record_getters=(lambda: 9,),
        )
    )

    assert flow.measure_kwargs["extra_record_columns"] == ("aux",)


def test_record_getters_extend_recorded_records_before_plot():
    flow = RecordingFlow(record_num=3)
    record_values = iter((5, 6))
    plot_values = iter((99, 100))
    seen_by_record = []
    seen_by_plot = []

    def on_record(records):
        seen_by_record.append(records)

    def update(plotobj, records):
        seen_by_plot.append(records)

    recipe = MeasurementRecipe(
        measure_mods=("I_source_sweep_dc", "V_sense_dc"),
        extra_record_columns=("aux",),
        extra_record_getters=(lambda: next(record_values),),
        plot=PlotRecipe(
            update=update,
            extra_getters=(lambda: next(plot_values),),
        ),
        on_record=on_record,
    )

    flow.run_recipe(recipe)

    assert flow.recorded == [(1, 2, 5), (3, 4, 6)]
    assert seen_by_record == [(1, 2, 5), (3, 4, 6)]
    assert seen_by_plot == [(1, 2, 5, 99), (3, 4, 6, 100)]


def test_mapped_plot_accepts_extra_getters():
    getter = lambda: 99

    plot = PlotModules.mapped_plot(
        init_args=(1, 1, 1),
        series=(
            PlotSeries(row=0, col=0, line=0, x_col=0, y_col=2),
        ),
        extra_getters=(getter,),
    )

    assert plot.extra_getters == (getter,)


def test_recipe_builders_pass_record_getters():
    getter = lambda: 1

    recipe = assemble_recipe(
        extra_record_columns=("aux",),
        extra_record_getters=(getter,),
    )
    built_recipe = RecipeBuilder().build(
        extra_record_columns=("aux",),
        extra_record_getters=(getter,),
    )

    assert recipe.extra_record_columns == ("aux",)
    assert recipe.extra_record_getters == (getter,)
    assert built_recipe.extra_record_columns == ("aux",)
    assert built_recipe.extra_record_getters == (getter,)


@pytest.mark.parametrize("records, expected", [
    ((1.5, 2.5), (1.5, 2.5)),
    (("1.5", "2.5"), (1.5, 2.5)),
    (("2026-11-01_12:00:00.123", "2.5"), ("2026-11-01_12:00:00.123", 2.5)),
    (("1.5", "category"), (1.5, "category")),
])
def test_mapped_plot_converts_strings_per_axis(records, expected):
    plot = PlotModules.mapped_plot(
        init_args=(1, 1, 1), series=[PlotSeries(0, 0, 0, 0, 1, "t", "X")],
        init_kwargs={"show": False}, saving_interval=None,
    )
    obj = DataManipulator(1)
    obj.live_plot_init(*plot.init_args, **plot.init_kwargs)
    plot.update(obj, records)
    trace = obj.go_f.data[0]
    assert list(trace.x) == [expected[0]]
    assert list(trace.y) == [expected[1]]


def test_mapped_plot_string_mode_does_not_leak_to_other_updates():
    options = {"max_points": 2}
    plot = PlotModules.mapped_plot(
        init_args=(1, 2, 1),
        series=[PlotSeries(0, 0, 0, 0, 1), PlotSeries(0, 1, 0, 2, 1)],
        init_kwargs={"show": False}, update_kwargs=options, saving_interval=None,
    )
    obj = DataManipulator(1)
    obj.live_plot_init(*plot.init_args, **plot.init_kwargs)
    for second in range(3):
        plot.update(obj, (f"2026-11-01_12:00:0{second}", "2.5", str(second)))
    assert list(obj.go_f.data[0].x) == ["2026-11-01_12:00:01", "2026-11-01_12:00:02"]
    assert list(obj.go_f.data[0].y) == [2.5, 2.5]
    assert list(obj.go_f.data[1].x) == [1., 2.]
    assert list(obj.go_f.data[1].y) == [2.5, 2.5]
    assert np.asarray(obj.go_f.data[1].x).dtype.kind == "f"
    assert options == {"max_points": 2}


@pytest.mark.parametrize("with_str", [False, True])
def test_mapped_plot_respects_explicit_string_mode(with_str):
    plot = PlotModules.mapped_plot(
        init_args=(1, 1, 1), series=[PlotSeries(0, 0, 0, 0, 1)],
        init_kwargs={"show": False}, update_kwargs={"with_str": with_str},
        saving_interval=None,
    )
    obj = DataManipulator(1)
    obj.live_plot_init(*plot.init_args, **plot.init_kwargs)
    if with_str:
        plot.update(obj, ("category", "2.5"))
        assert list(obj.go_f.data[0].y) == ["2.5"]
    else:
        with pytest.raises(ValueError, match="could not convert string to float"):
            plot.update(obj, ("category", "2.5"))


def test_mapped_plot_does_not_hide_invalid_line_shapes():
    plot = PlotModules.mapped_plot(
        init_args=(1, 1, 1), series=[PlotSeries(0, 0, 0, 0, 1)],
        init_kwargs={"show": False}, saving_interval=None,
    )
    obj = DataManipulator(1)
    obj.live_plot_init(*plot.init_args, **plot.init_kwargs)
    with pytest.raises(ValueError, match="lengths"):
        plot.update(obj, ("category", [1., 2.]))
    assert len(obj.go_f.data[0].x) == 0
