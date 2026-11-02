import json
from dataclasses import asdict
from inspect import GEN_CLOSED, getgeneratorstate
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from pyomnix import DataManipulator
from pyflexlab.equip_wrapper.simulated import SimMeter
from pyflexlab.file_organizer import FileOrganizer
from pyflexlab.measure_manager import MeasureManager, ReadWindow
from pyflexlab.measure_flow import MeasureFlow, MeasurementRecipe, RecordSource, PulseStep, PULSE_COLUMNS
from pyflexlab.recipe_builders import MeasureModules, RecipeOptions, assemble_recipe, PlotModules, PlotSeries


@pytest.fixture
def flow():
    # Offline runner; individual fixtures supply project paths and instruments.
    instance = object.__new__(MeasureFlow)
    instance._csv_fast_writer = None
    yield instance
    instance.record_finalize()


@pytest.fixture
def scan_hardware(flow, tmp_path, monkeypatch):
    """Exercise real preparation/source/sense/naming code with instrument I/O replaced."""
    template = Path(__file__).resolve().parents[1] / "pyflexlab/templates/measure_types.json"
    monkeypatch.setattr(FileOrganizer, "measure_types_json", json.loads(template.read_text()))
    flow._out_database_dir_proj = tmp_path
    monkeypatch.setattr(flow, "add_measurement", lambda *a: None)
    state, events = {}, []
    sources = [SimMeter(), SimMeter()]
    sense = SimMeter()
    for index, meter in enumerate(sources):
        def output(value, *, index=index, **kwargs):
            state[index] = value
            events.append(f"source{index}")
            assert kwargs["compliance"] == .001
            return value
        meter.setup = Mock()
        meter.uni_output = Mock(side_effect=output)
        meter.ramp_output = Mock()
        meter.output_switch = Mock()
    def read(type_str):
        events.append("read")
        return state[0] + 10 * state[1]
    sense.setup = Mock()
    sense.sense = Mock(side_effect=read)
    recipe = assemble_recipe(
        *(MeasureModules.sweep_voltage_source(.2, .1, high=1, low=0,
            sweepmode="manual", meter=meter, compliance="1mA") for meter in sources),
        MeasureModules.voltage_sense(high=1, low=0, meter=sense, ac_dc="dc"),
        options=RecipeOptions(source_wait=0, appendix_str="scan"), shutdown=sources,
    )
    return SimpleNamespace(recipe=recipe, sources=sources, sense=sense, events=events)


@pytest.fixture
def ordinary_recipe(flow, tmp_path, monkeypatch):
    """Use the real file/recipe pipeline, replacing only instrument operations."""
    template = Path(__file__).resolve().parents[1] / "pyflexlab/templates/measure_types.json"
    monkeypatch.setattr(FileOrganizer, "measure_types_json", json.loads(template.read_text()))
    monkeypatch.setattr(flow, "get_filepath", lambda *a, **kw: tmp_path / (
        "data.png" if kw.get("plot") else "data.csv"))
    monkeypatch.setattr(flow, "add_measurement", lambda *a: None)
    monkeypatch.setattr(flow, "sense_apply", lambda *a, **kw: iter([.25, .5]))
    return MeasurementRecipe(
        ("V_sense_dc",), args=("", 1, 0), wrapper_lst=[SimpleNamespace(setup=Mock())],
        measure_kwargs={"with_timer": False},
    )


def test_repeated_recipe_backs_up_csv_only(flow, ordinary_recipe, tmp_path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    previous_csv = None
    for run in range(3):
        values = [run + .25, run + .5]
        monkeypatch.setattr(flow, "sense_apply", lambda *a, **kw: iter(values))
        flow.run_recipe(ordinary_recipe)
        assert pd.read_csv(csv_path).V.tolist() == values
        if run:
            assert (tmp_path / "data.csv.bak").read_bytes() == previous_csv
        previous_csv = csv_path.read_bytes()
    assert ordinary_recipe.wrapper_lst[0].setup.call_count == 3
    assert {p.name for p in tmp_path.iterdir()} == {"data.csv", "data.csv.bak"}


def test_custom_source_ignores_companion_json_and_preserves_existing_csv(flow, tmp_path):
    source = RecordSource(Mock(return_value=iter([(1.,)])), ("V",), tmp_path / "data.csv")
    companion = tmp_path / "data.metadata.json"
    companion.write_text('{"old": true}', encoding="utf-8")
    original = companion.read_bytes()
    recipe = MeasurementRecipe((), record_source=source)
    flow.run_recipe(recipe)
    csv_content = source.file_path.read_bytes()
    with pytest.raises(FileExistsError):
        flow.run_recipe(recipe)
    source.factory.assert_called_once_with()
    assert source.file_path.read_bytes() == csv_content
    assert companion.read_bytes() == original
    assert {p.name for p in tmp_path.iterdir()} == {"data.csv", "data.metadata.json"}


@pytest.mark.parametrize("dtype", [np.int64, np.float32])
def test_numpy_calibration_scan_records_csv_only(flow, tmp_path, dtype, scan_hardware):
    plan = MeasureManager.create_scan_plan([0, .1], [0], offset=np.array([1, 2], dtype=dtype))
    recipe = flow.scan_recipe(plan, scan_hardware.recipe, reading_columns=("V",))
    mea_dict = flow.run_recipe(recipe)
    frame = pd.read_csv(mea_dict["file_path"])
    np.testing.assert_allclose(frame.V_top, [1., 1.1, 1.])
    np.testing.assert_allclose(frame.V_bottom, [2., 2., 2.])
    assert len(frame) == len(plan.points)
    assert list(tmp_path.rglob("*.json")) == []
    assert list(tmp_path.rglob("*.csv")) == [mea_dict["file_path"]]


def test_affine_calibration_roundtrip_and_boundaries():
    matrix, offset = [[-1, 1], [17.57, 17.57]], [-.4, -18]
    x, y = np.meshgrid([-.5, 0, .5], [0, 1])
    a, b = MeasureManager.coordinate_transform(x, y, matrix=matrix, offset=offset)
    xx, yy = MeasureManager.coordinate_transform(a, b, matrix=matrix, offset=offset, inverse=True)
    np.testing.assert_allclose(xx, x, atol=1e-14)
    np.testing.assert_allclose(yy, y, atol=1e-14)
    with pytest.raises(ValueError, match="limits"):
        MeasureManager.coordinate_transform(10, 10, matrix=matrix, limits=[[-1, 1], [-2, 2]])
    with pytest.raises(np.linalg.LinAlgError):
        MeasureManager.coordinate_transform(0, 0, matrix=[[1, 1], [2, 2]])


@pytest.mark.parametrize("start,first,directions", [("low", [0, 1, 2, 1, 0], [1, 1, 1, -1, -1]),
                                                    ("high", [2, 1, 0, 1, 2], [-1, -1, -1, 1, 1])])
def test_scan_plan_roundtrip_repeats_and_manual_tables(start, first, directions):
    plan = MeasureManager.create_scan_plan([0, 1, 2], [10, 20], repeats=3, start=start)
    assert len(plan.points) == 30
    assert plan.points.scan_id.nunique() == 6
    assert plan.points.cycle_id.nunique() == 3
    assert list(plan.points.ix[:5]) == first
    assert list(plan.points.direction[:5]) == directions
    assert list(plan.points.point_id) == list(range(30))
    assert len(plan.sweep_tables) == 2 and len(plan.sweep_tables[0]) == 30
    assert plan.grid().shape == (2, 3)
    assert plan.metadata["planned_points"] == len(plan.points)


def test_constrained_scan_reports_omissions_without_clipping():
    with pytest.raises(ValueError, match="outside"):
        MeasureManager.create_scan_plan([-1, 0, 1], [0, 3], limits=[[-1, 1], [-1, 1]])
    plan = MeasureManager.create_scan_plan([-1, 0, 1], [0, 3], limits=[[-1, 1], [-1, 1]],
                                           out_of_bounds="omit")
    assert plan.metadata["skipped_rows"] == [1]
    assert not plan.valid[1].any()
    assert set(plan.points.iy) == {0}
    assert list(plan.points.V_top) == [-1, 0, 1, 0, -1]


def test_real_runner_records_scan_and_flushes_grid(flow, tmp_path, scan_hardware, monkeypatch):
    plan = MeasureManager.create_scan_plan([0, .1, .2], [0, .1])
    up, down = plan.grid(), plan.grid()
    columns = (*plan.columns, "elapsed_s", "V_read")
    direction_index = columns.index("direction")
    plot = PlotModules.mapped_plot(
        init_args=(1, 2, 1), record_columns=columns, saving_interval=None,
        refresh_interval_s=3600, init_kwargs={"show": False},
        series=tuple(PlotSeries(0, col, 0, "ix", "iy", z_col="V_read", grid=grid,
                                when=lambda record, sign=sign: record[direction_index] == sign,
                                grid_update_kwargs={"indices": True})
                     for col, (grid, sign) in enumerate([(up, 1), (down, -1)])))
    plot.plotobj = DataManipulator(1)
    scan_hardware.recipe.plot = plot
    recipe = flow.scan_recipe(plan, scan_hardware.recipe, reading_columns=("V_read",),
                              wait=lambda: scan_hardware.events.append("wait"))
    prepare = Mock(wraps=flow.get_measure_dict)
    monkeypatch.setattr(flow, "get_measure_dict", prepare)
    assert not scan_hardware.events
    assert recipe.record_source is None
    mea_dict = flow.run_recipe(recipe)
    assert prepare.called
    assert prepare.call_args_list[0].kwargs["if_combine_gen"] is False
    assert mea_dict["swp_idx"] == [columns.index(name) for name in plan.source_names]
    frame = pd.read_csv(mea_dict["file_path"])
    assert len(frame) == len(plan.points)
    np.testing.assert_allclose(frame[list(plan.columns)], plan.points)
    np.testing.assert_allclose(frame.V_read, plan.points.x + 10 * plan.points.y)
    assert scan_hardware.events == ["source0", "source1", "wait", "read"] * len(frame)
    for index, meter in enumerate(scan_hardware.sources):
        meter.setup.assert_called_once_with(function="source", source_type="V")
        meter.ramp_output.assert_called_once()
        np.testing.assert_allclose([call.args[0] for call in meter.uni_output.call_args_list],
                                   plan.sweep_tables[index])
        meter.output_switch.assert_called_once_with("off")
    assert tuple(frame.columns) == columns
    assert up.complete
    assert np.isnan(down.z[:, -1]).all()  # The turnaround was sampled once, on up.
    np.testing.assert_allclose(plot.plotobj.go_f.data[0].z, up.z, equal_nan=True)
    np.testing.assert_allclose(plot.plotobj.go_f.data[1].z, down.z, equal_nan=True)
    assert "swpmodemanual" in mea_dict["file_path"].name
    assert mea_dict["file_path"].name.endswith("-scan.csv")
    assert list(tmp_path.rglob("*.json")) == []
    assert flow._csv_fast_writer is None


@pytest.mark.parametrize("mode,start", [("forward", "high"), ("roundtrip", "low")])
def test_scan_recipe_preserves_order_hooks_and_can_run_again(flow, scan_hardware, mode, start):
    plan = MeasureManager.create_scan_plan([0, .1, .2], [0, .1, 1], mode=mode,
        start=start, repeats=2, limits=[[-.1, .3], [-.1, .2]], out_of_bounds="omit")
    original = scan_hardware.recipe
    prepared, measured, recorded = [], [], []
    original.after_prepare = lambda mea: prepared.append(tuple(mea["swp_idx"]))
    original.on_measure = measured.append
    original.on_record = recorded.append
    original.extra_record_columns = ("temperature",)
    original.extra_record_getters = (lambda: 4.2,)
    recipe = flow.scan_recipe(plan, original, reading_columns=("V_read",))
    assert original.measure_kwargs == {"source_wait": 0, "appendix_str": "scan"}
    previous = None
    for _ in range(2):
        mea = flow.run_recipe(recipe)
        frame = pd.read_csv(mea["file_path"])
        np.testing.assert_allclose(frame[list(plan.columns)], plan.points)
        np.testing.assert_allclose(frame.temperature, 4.2)
        if previous is not None:
            assert mea["file_path"].with_suffix(".csv.bak").read_bytes() == previous
        previous = mea["file_path"].read_bytes()
    assert prepared == [(8, 9), (8, 9)]
    assert len(measured) == len(recorded) == 2 * len(plan.points)
    assert all(len(row) == len(plan.columns) + 2 for row in measured)
    assert all(len(row) == len(plan.columns) + 3 for row in recorded)


def test_scan_recipe_flattens_multiple_and_ac_senses(flow, scan_hardware):
    plan = MeasureManager.create_scan_plan([.1], [.2], mode="forward")
    recipe = scan_hardware.recipe
    ac_meter = SimMeter()
    ac_meter.setup = Mock()
    ac_meter.sense = Mock(return_value=(1., 2., 3., 4.))
    recipe.measure_mods += ("V_sense_ac",)
    recipe.args += ("lockin", 1, 0)
    recipe.wrapper_lst.append(ac_meter)
    columns = ("V_read", "X", "Y", "R", "phase")
    result = flow.run_recipe(flow.scan_recipe(plan, recipe, reading_columns=columns))
    frame = pd.read_csv(result["file_path"])
    assert tuple(frame.columns) == (*plan.columns, "elapsed_s", *columns)
    np.testing.assert_allclose(frame[list(columns)], [[2.1, 1., 2., 3., 4.]])


@pytest.mark.parametrize("columns", [(), ("a", "b"), ("point_id",)])
def test_scan_recipe_rejects_invalid_columns_before_instrument_io(flow, scan_hardware, columns):
    plan = MeasureManager.create_scan_plan([0, .1], [0])
    with pytest.raises(ValueError, match="column"):
        flow.scan_recipe(plan, scan_hardware.recipe, reading_columns=columns)
    for meter in scan_hardware.sources:
        meter.setup.assert_not_called()
        meter.uni_output.assert_not_called()


@pytest.mark.parametrize("bad_generator", [0, 2])
def test_scan_records_rejects_short_generators_and_closes_children(flow, bad_generator):
    plan = MeasureManager.create_scan_plan([0, .1], [0])
    generators = [iter(values for values in table) for table in (*plan.sweep_tables, [1.] * 3)]
    generators[bad_generator] = (value for value in [0.])
    rows = flow.scan_records(plan, generators)
    next(rows)
    with pytest.raises(RuntimeError, match="ended before point 1"):
        next(rows)
    assert all(getgeneratorstate(generator) == GEN_CLOSED for generator in generators)


def test_scan_records_closes_children_on_manual_stop(flow):
    plan = MeasureManager.create_scan_plan([0, .1], [0])
    generators = [(value for value in table) for table in (*plan.sweep_tables, [1.] * 3)]
    rows = flow.scan_records(plan, generators)
    next(rows)
    rows.close()
    assert all(getgeneratorstate(generator) == GEN_CLOSED for generator in generators)


def test_scan_recipe_rejects_nonfinite_reads(flow, scan_hardware, tmp_path):
    plan = MeasureManager.create_scan_plan([0, .1], [0])
    scan_hardware.sense.sense.side_effect = [1., float("nan")]
    recipe = flow.scan_recipe(plan, scan_hardware.recipe, reading_columns=("V_read",))
    with pytest.raises(ValueError, match="finite"):
        flow.run_recipe(recipe)
    assert len(pd.read_csv(next(tmp_path.rglob("*.csv")))) == 1
    assert flow._csv_fast_writer is None


@pytest.mark.parametrize("ac_dc", ["ac", "dc"])
def test_manual_single_point_sweep_still_ramps(flow, scan_hardware, ac_dc):
    meter = scan_hardware.sources[0]
    rows = flow.source_sweep_apply("V", ac_dc, meter, max_value="100mV",
        step_value="10mV", compliance="1mA", freq=17 if ac_dc == "ac" else None,
        sweepmode="manual", sweep_table=["100mV"], source_wait=0)
    assert list(rows) == [.1]
    assert meter.ramp_output.call_args.args == ("volt", .1)
    meter.uni_output.assert_called_once()


@pytest.mark.parametrize("table,allow_large_jump", [(None, False), ([], False), ([], True)])
def test_manual_empty_table_is_rejected(flow, scan_hardware, table, allow_large_jump):
    meter = scan_hardware.sources[0]
    with pytest.raises(ValueError, match="nonempty"):
        list(flow.source_sweep_apply("V", "dc", meter, max_value=.1, step_value=.1,
            compliance="1mA", sweepmode="manual", sweep_table=table,
            allow_large_jump=allow_large_jump, source_wait=0))
    meter.ramp_output.assert_not_called()
    meter.uni_output.assert_not_called()


def test_scan_keeps_manual_jump_validation(flow, scan_hardware):
    plan = MeasureManager.create_scan_plan([0, 1], [0])
    recipe = flow.scan_recipe(plan, scan_hardware.recipe, reading_columns=("V_read",))
    with pytest.raises(ValueError, match="large jump"):
        flow.run_recipe(recipe)
    scan_hardware.sources[0].ramp_output.assert_not_called()
    scan_hardware.sources[0].uni_output.assert_not_called()


@pytest.mark.parametrize("panel_count", [1, 2])
def test_throttled_grid_updates_forward_options_and_flush_every_sample(monkeypatch, panel_count):
    import pyflexlab.recipe_builders as module
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    plan = MeasureManager.create_scan_plan([0, 1], [0])
    grids = [plan.grid() for _ in range(panel_count)]
    plot = PlotModules.mapped_plot(
        init_args=(1, panel_count, 1), saving_interval=None, refresh_interval_s=10,
        init_kwargs={"show": False}, update_kwargs={"with_str": False},
        series=[PlotSeries(0, col, 0, 0, 1, z_col=2, grid=grid,
                           grid_update_kwargs={"indices": True}) for col, grid in enumerate(grids)],
    )
    plotobj = DataManipulator(1)
    plotobj.live_plot_init(*plot.init_args, **plot.init_kwargs)
    update = Mock(wraps=plotobj.live_plot_update)
    monkeypatch.setattr(plotobj, "live_plot_update", update)
    plot.update(plotobj, (0, 0, 1.))
    clock.now = 1.
    plot.update(plotobj, (1, 0, 2.))
    assert update.call_count == 1
    assert all(grid.complete for grid in grids)
    assert np.isnan(plotobj.go_f.data[0].z[0][1])
    plot.flush(plotobj)
    assert update.call_count == 2
    assert all(call.kwargs == {"with_str": False} for call in update.call_args_list)
    for trace in plotobj.go_f.data:
        np.testing.assert_allclose(trace.z, [[1., 2.]])
    plot.flush(plotobj)
    assert update.call_count == 2  # Nothing pending after a successful flush.


def test_failed_scan_flushes_grid_before_stopping_plot(flow, tmp_path, monkeypatch):
    grid = MeasureManager.create_scan_plan([0, 1], [0]).grid()
    plotobj = DataManipulator(1)
    final_snapshots = []
    monkeypatch.setattr(plotobj, "stop_saving", lambda: final_snapshots.append(
        np.array(plotobj.go_f.data[0].z, dtype=float)))
    plot = PlotModules.mapped_plot(
        init_args=(1, 1, 1), plotobj=plotobj, saving_interval=None,
        refresh_interval_s=3600, init_kwargs={"show": False},
        series=[PlotSeries(0, 0, 0, 0, 1, z_col=2, grid=grid)],
    )
    def rows():
        yield 0., 0., 1.
        yield 1., 0., 2.
        raise RuntimeError("read failed")
    source = RecordSource(rows, ("x", "y", "V"), tmp_path / "failed.csv")
    shutdown = Mock(spec=lambda: None)
    with pytest.raises(RuntimeError, match="read failed"):
        flow.run_recipe(MeasurementRecipe((), record_source=source, plot=plot, shutdown=(shutdown,)))
    np.testing.assert_allclose(final_snapshots, [[[1., 2.]]])
    assert len(pd.read_csv(source.file_path)) == 2
    shutdown.assert_called_once_with()
    assert flow._csv_fast_writer is None


@pytest.mark.parametrize("use_grid", [False, True])
def test_plot_options_are_validated_by_the_shared_live_update(use_grid):
    grid = MeasureManager.create_scan_plan([0, 1], [0]).grid() if use_grid else None
    plot = PlotModules.mapped_plot(
        init_args=(1, 1, 1), init_kwargs={"show": False}, update_kwargs={"max_points": 0},
        series=[PlotSeries(0, 0, 0, 0, 1, z_col=2 if use_grid else None, grid=grid)],
    )
    plotobj = DataManipulator(1)
    plotobj.live_plot_init(*plot.init_args, **plot.init_kwargs)
    with pytest.raises(ValueError, match="max_points"):
        plot.update(plotobj, (0., 0., 1.))


@pytest.mark.parametrize("failure", ["source", "wait", "read"])
def test_runner_failure_closes_record_and_runs_shutdown(flow, tmp_path, scan_hardware, failure):
    plan = MeasureManager.create_scan_plan([0, .1], [0])
    wait = Mock()
    if failure == "source":
        scan_hardware.sources[0].uni_output.side_effect = [0., RuntimeError("source failed")]
        scan_hardware.sense.sense.side_effect = [1.]
    elif failure == "wait":
        wait.side_effect = [None, RuntimeError("wait failed")]
    else:
        scan_hardware.sense.sense.side_effect = [1., RuntimeError("read failed")]
    recipe = flow.scan_recipe(plan, scan_hardware.recipe, reading_columns=("V_read",), wait=wait)
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        flow.run_recipe(recipe)
    assert len(pd.read_csv(next(tmp_path.rglob("*.csv")))) == 1
    for meter in scan_hardware.sources:
        meter.output_switch.assert_called_once_with("off")
    assert flow._csv_fast_writer is None






def test_filename_index_preserves_repeated_parameter_values(tmp_path):
    files = [tmp_path / f"scan-n1-repeat{i}.csv" for i in range(2)]
    for file in files:
        file.write_text("V\n1\n")
    table = FileOrganizer.index_measurements(
        files, pattern=r"scan-n(?P<n>\d+)-repeat(?P<repeat>\d+)",
    )
    assert len(table) == 2 and list(table.n) == ["1", "1"]
    assert list(table.repeat) == ["0", "1"]
    assert list(table.columns) == ["path", "bytes", "n", "repeat"]


def test_wait_requires_consecutive_passing_readings_and_has_deadline(monkeypatch):
    import pyflexlab.measure_manager as module
    class Clock:
        now = 0.
        def monotonic(self):
            return self.now
        def sleep(self, seconds):
            self.now += seconds
    clock = Clock()
    monkeypatch.setattr(module, "time", clock)
    readings = iter([0, 0, 2, 0, 0, 0])
    result = MeasureManager.wait_until(lambda: next(readings), lambda x: x < 1,
                                       timeout=2, interval=.1, consecutive=3)
    assert result.samples == 6
    with pytest.raises(TimeoutError):
        MeasureManager.wait_until(lambda: 2, lambda x: x < 1, timeout=.15, interval=.1)


def test_stable_read_window_keeps_minimum_samples_and_rejects_nan():
    readings = iter([1, 1.1, 1.1, 1.1])
    rows = list(MeasureManager.read_window(lambda: next(readings),
                ReadWindow(until_stable=True, min_samples=4, consecutive=2,
                           atol=0, rtol=0, timeout_s=1, interval_s=0)))
    assert len(rows) == 4 and [row[0] for row in rows] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="nonfinite"):
        list(MeasureManager.read_window(lambda: float("nan"), ReadWindow(samples=1)))


def test_explicit_non_visa_initialization_keeps_default_behavior(monkeypatch):
    import pyflexlab.measure_manager as module
    monkeypatch.setattr(FileOrganizer, "__init__", lambda self, *a, **k: None)
    calls = []
    manager = object()
    monkeypatch.setattr(module.pyvisa, "ResourceManager", lambda: calls.append(True) or manager)
    offline = MeasureFlow("offline", connect_visa=False)
    assert not calls and offline._rm is None
    with pytest.raises(RuntimeError, match="disabled"):
        offline.list_visa_resources()
    default = MeasureFlow("default")
    assert calls == [True] and default._rm is manager
