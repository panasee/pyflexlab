import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyflexlab.file_organizer import FileOrganizer
from pyflexlab.equip_wrapper import Wrapper6221
from pyflexlab.measure_manager import MeasureManager
from pyflexlab.measure_flow import MeasureFlow, MeasurementRecipe, PlotRecipe
from pyflexlab.recipe_builders import MeasureModules, RecipeOptions, assemble_recipe


class PulseSource(Wrapper6221):
    def __init__(self):
        self.meter = "fake6221"
        self.calls = []

    def pulse_output(self, **kwargs):
        self.calls.append(kwargs)

    def setup(self, **kwargs):
        raise AssertionError("pulse must not enter ordinary source setup")


class Sense:
    def setup(self, **kwargs):
        pass


@pytest.fixture
def flow(tmp_path, monkeypatch):
    template = Path(__file__).resolve().parents[1] / "pyflexlab/templates/measure_types.json"
    monkeypatch.setattr(FileOrganizer, "measure_types_json", json.loads(template.read_text()))
    obj = object.__new__(MeasureFlow)
    obj._csv_fast_writer = None
    monkeypatch.setattr(obj, "get_filepath", lambda *a, **kw: tmp_path / ("plot.png" if kw.get("plot") else "data.csv"))
    monkeypatch.setattr(obj, "add_measurement", lambda *a: None)
    monkeypatch.setattr(obj, "sense_apply", lambda *a, **kw: iter([0.25] * 10))
    yield obj
    obj.record_finalize()


FIXED_ARGS = ("0A", "1mA", "1ms", "10Hz", "2")
SWEEP_ARGS = (0, "2mA", "1mA", "manual", "1ms", "10Hz", 2)
TABLE = [[0, "1mA", "1ms", "10Hz", 2], [0, "2mA", "2ms", "20Hz", 3]]


@pytest.mark.parametrize("timer", [False, True])
def test_fixed_pulse_five_columns_and_sense(flow, timer):
    source = PulseSource()
    result = flow.get_measure_dict(
        ("I_source_fixed_pulse", "V_sense_dc"), *FIXED_ARGS, "", 1, 0,
        wrapper_lst=[source, Sense()], compliance_lst=[5], with_timer=timer,
    )
    row = next(result["gen_lst"])
    assert result["record_num"] == 6 + timer
    assert row[int(timer):] == pytest.approx([0, .001, .001, 10, 2, .25])
    flow.record_update(result["file_path"], result["record_num"], row, force_write=True)
    saved = pd.read_csv(result["file_path"])
    assert list(saved.columns) == (["time"] if timer else []) + [
        "I_pulse_bot", "I_pulse_top", "pulse_width", "pulse_frequency", "pulse_count", "V"
    ]
    assert saved.shape == (1, 6 + timer)
    assert len(source.calls) == 1


@pytest.mark.parametrize("table_factory", [list, np.array, pd.DataFrame])
@pytest.mark.parametrize("timer", [False, True])
def test_manual_pulse_flattens_five_columns(flow, table_factory, timer):
    result = flow.get_measure_dict(
        ("I_source_sweep_pulse",), *SWEEP_ARGS,
        wrapper_lst=[PulseSource()], compliance_lst=[5],
        sweep_tables=[table_factory(TABLE)], with_timer=timer,
    )
    rows = list(result["gen_lst"])
    assert result["record_num"] == 5 + timer
    assert result["swp_idx"] == [1 + timer]
    assert len(rows) == 2
    assert rows[1][int(timer):] == pytest.approx([0, .002, .002, 20, 3])


def test_later_sweep_index_accounts_for_pulse_width(flow, monkeypatch):
    monkeypatch.setattr(flow, "ext_sweep_apply", lambda *a, **kw: iter([.1, .2]))
    result = flow.get_measure_dict(
        ("I_source_sweep_pulse", "B_sweep"), *SWEEP_ARGS, 0, .2, .1, "min-max",
        wrapper_lst=[PulseSource()], compliance_lst=[5], sweep_tables=[TABLE],
    )
    assert result["swp_idx"] == [2, 6]
    assert len(next(result["gen_lst"])) == result["record_num"] == 7


def test_runner_csv_and_plot_with_extra_column(flow, monkeypatch):
    seen = []
    plotted = []

    class Plot:
        def stop_saving(self):
            pass

    monkeypatch.setattr(flow, "_init_recipe_plot", lambda *a: Plot())
    result = flow.run_recipe(MeasurementRecipe(
        measure_mods=("I_source_sweep_pulse",), args=SWEEP_ARGS,
        wrapper_lst=[PulseSource()], compliance_lst=[5],
        measure_kwargs={"sweep_tables": [TABLE]},
        extra_record_columns=("aux",), extra_record_getters=(lambda: 9,),
        plot=PlotRecipe(update=lambda plot, row: plotted.append(row)),
        on_record=lambda row: seen.append(row),
    ))
    flow.record_finalize()
    saved = pd.read_csv(result["file_path"])
    assert saved.shape == (2, 7)
    assert all(len(row) == result["record_num"] for row in seen)
    assert plotted == seen
    assert list(saved["aux"]) == [9, 9]


def test_manual_columns_survive_tuple_table_conversion(flow):
    columns = ["bot", "top", "width", "freq", "count"]
    result = flow.get_measure_dict(
        ("I_source_sweep_pulse",), *SWEEP_ARGS,
        wrapper_lst=[PulseSource()], compliance_lst=[5], with_timer=False,
        sweep_tables=(TABLE,), manual_record_columns=columns,
    )
    assert result["record_num"] == 5
    flow.record_finalize()
    assert list(pd.read_csv(result["file_path"]).columns) == columns


def test_incorrect_manual_column_count_fails(flow):
    result = flow.get_measure_dict(
        ("I_source_fixed_pulse",), *FIXED_ARGS,
        wrapper_lst=[PulseSource()], compliance_lst=[5],
        manual_record_columns=["time", "current"],
    )
    row = next(result["gen_lst"])
    with pytest.raises(ValueError, match="The number of columns does not match"):
        flow.record_update(
            result["file_path"], result["record_num"], row,
        )


def test_automatic_pulse_sweep(flow):
    result = flow.get_measure_dict(
        ("I_source_sweep_pulse",), 0, .002, .001, "min-max", .001, 10, 2,
        wrapper_lst=[PulseSource()], compliance_lst=[5], with_timer=False,
    )
    rows = list(result["gen_lst"])
    assert [row[1] for row in rows] == pytest.approx([0, .001, .002])
    assert all(len(row) == result["record_num"] == 5 for row in rows)


def test_ac_sense_and_uncombined_generators(flow, monkeypatch):
    monkeypatch.setattr(flow, "sense_apply", lambda *a, **kw: iter([(1, 2, 3, 4)]))
    result = flow.get_measure_dict(
        ("I_source_fixed_pulse", "V_sense_ac"), *FIXED_ARGS, "", 1, 0,
        wrapper_lst=[PulseSource(), Sense()], compliance_lst=[5],
        with_timer=False, if_combine_gen=False,
    )
    parts = [next(gen) for gen in result["gen_lst"]]
    assert [len(part) for part in parts] == [5, 4]
    assert result["record_num"] == 9


def test_manual_columns_not_modified_when_adding_extra(flow):
    columns = ["bot", "top", "width", "freq", "count"]
    result = flow.get_measure_dict(
        ("I_source_fixed_pulse",), *FIXED_ARGS,
        wrapper_lst=[PulseSource()], compliance_lst=[5], with_timer=False,
        manual_record_columns=columns, extra_record_columns=["aux"],
    )
    assert len(columns) == 5
    assert result["record_num"] == 6


def test_fixed_pulse_builder_prepares_real_generator(flow):
    source = PulseSource()
    recipe = assemble_recipe(MeasureModules.fixed_current_pulse(
        "0A", "1mA", meter=source, compliance="5V",
        pulse_width="1ms", freq="10Hz", pulse_count=2,
    ))
    assert source.calls == []
    result = flow.prepare_recipe(recipe)
    row = next(result["gen_lst"])
    assert row[1:] == pytest.approx([0, .001, .001, 10, 2])
    assert result["record_num"] == len(row) == 6
    assert source.calls[0]["compliance"] == "5V"


@pytest.mark.parametrize("mode", ["manual", "min-max"])
def test_sweep_pulse_builder_runs_recipe(flow, mode):
    source = PulseSource()
    module = MeasureModules.sweep_current_pulse(
        0, "2mA", "1mA", sweepmode=mode, meter=source, compliance="5V",
        pulse_width="1ms", freq="10Hz", pulse_count=2,
    )
    result = flow.run_recipe(assemble_recipe(
        module,
        options=RecipeOptions(sweep_tables=[TABLE] if mode == "manual" else None),
    ))
    flow.record_finalize()
    saved = pd.read_csv(result["file_path"])
    expected = [.001, .002] if mode == "manual" else [0, .001, .002]
    assert saved.shape == (len(expected), 6)
    assert list(saved["I_pulse_top"]) == pytest.approx(expected)
    assert len(source.calls) == len(expected)


@pytest.mark.parametrize("sweep", [False, True])
def test_pulse_builders_reject_other_meters(sweep):
    builder = MeasureModules.sweep_current_pulse if sweep else MeasureModules.fixed_current_pulse
    with pytest.raises(ValueError, match="6221"):
        builder(
            *(0, .002, .001) if sweep else (0, .001),
            **({"sweepmode": "min-max"} if sweep else {}),
            meter=Sense(), compliance=5, pulse_width=.001, freq=10, pulse_count=2,
        )
