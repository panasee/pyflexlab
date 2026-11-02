"""
High-level measurement flow interface.

The legacy method collection lives in measure_flow_old.py. This module keeps
those public methods available while adding a smaller recipe/runner layer for
new measurement flows.

HOOKS:
after_prepare(PrepareHook)
on_measure(MeasureHook)
on_record(RecordHook)
shutdown(ShutdownTarget)

notes for Hooks:
1. `on_measure` and `on_record` hooks are similar, only difference is
    `on_measure` follows measurement(@`for i in mea_gen`, when activating generator)
    `record_update` follows record writing(record_update)
2. Use `on_measure` or `on_record` for changes during measurement (like start varying); use `after_prepare` for extra configuration before measurement (like change settings not in `get_measure_dict`)
"""

from __future__ import annotations

import time
import math
from contextlib import ExitStack
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence
from functools import wraps

from pyomnix.data_process import DataManipulator
from pyomnix.omnix_logger import get_logger
from pyomnix.utils import combined_generator_list

from pyflexlab.equip_wrapper import Meter, SourceMeter, WrapperSR830, WrapperSR860
from pyflexlab.constants import _validate_nonnegative
from pyflexlab.measure_manager import ReadWindow, ScanPlan
from pyflexlab.measure_flow_old import MeasureFlow as LegacyMeasureFlow

logger = get_logger(__name__)

RecordTuple = tuple[Any, ...]  # actual measured values tuple from the whole generator
PlotUpdate = Callable[[DataManipulator, RecordTuple], None]
PlotFlush = Callable[[DataManipulator], None]
PlotGetter = Callable[[], Any]
RecordGetter = Callable[[], Any]
RecordHook = Callable[[RecordTuple], None]
MeasureHook = Callable[[RecordTuple], None]
PrepareHook = Callable[[dict[str, Any]], None]
ShutdownTarget = SourceMeter | Callable[[], None]

def noop(*args, **kwargs):
    "used for None case"
    pass

def add_after(func: Callable | None, action: Callable):
    # decorator for add action to func, both return None
    base_func = func or noop
    @wraps(base_func)
    def wrapper(*args, **kwargs) -> None:
        base_func(*args, **kwargs)
        action(*args, **kwargs)
    return wrapper

def default_time_update(plotobj: DataManipulator, records: RecordTuple) -> None:
    """
    used when no manual appointed plot functions, plot only first two cols 
    """
    logger.validate(
        len(records) >= 2, "length of records is less than 2, cannot draw figures"
    )
    plotobj.live_plot_update(
        0,
        0,
        0,
        [records[0]],
        [records[1]],
        incremental=True,
    )


@dataclass(slots=True)
class PlotRecipe:
    """Plot configuration for a recipe runner.
    keep all default to get basic testing config"""

    enabled: bool = True
    plotobj: Optional[DataManipulator] = None
    # args & kwargs are all directly fed into live_plot_init
    init_args: tuple[Any, ...] = (1,1,1)
    init_kwargs: dict[str, Any] = field(default_factory=dict)

    update: PlotUpdate = default_time_update
    # Values appended only for plot.update; they are not recorded to CSV.
    extra_getters: Sequence[PlotGetter] = ()
    saving_interval: float | None = 7
    inline_jupyter: Optional[bool] = None
    # Flush pending display updates before stop_saving takes its final snapshot.
    flush: PlotFlush | None = None


@dataclass(frozen=True)
class PulseStep: # single pulse event
    value: float  # Write amplitude in the units expected by apply
    width_s: float  # Hold time in seconds after apply returns
    cycle_id: int = 0  # User-assigned cycle identifier
    settings: dict[str, float] = field(default_factory=dict)  # Named source setpoints appended on request

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            logger.raise_error("pulse value must be finite", ValueError, include_traceback=False)
        _validate_nonnegative(self.width_s, "width_s")
        if not isinstance(self.cycle_id, int) or self.cycle_id < 0:
            logger.raise_error("cycle_id must be a nonnegative integer", ValueError, include_traceback=False)
        if any(not isinstance(k, str) or not k or not math.isfinite(v) for k, v in self.settings.items()):
            logger.raise_error("pulse settings must have nonempty names and finite values", ValueError, include_traceback=False)
        object.__setattr__(self, "settings", dict(self.settings))


PULSE_COLUMNS = ("cycle_id", "pulse_id", "sample_id", "phase", "pulse_value",
                 "pulse_width_s", "read_bias", "elapsed_s", "time_since_pulse_s", "I_read")  # Ordered ten-column row schema


@dataclass(slots=True)
class RecordSource:
    """Custom row factory. Columns include time, if wanted; files must be new.

    The factory is called only after preparation and plot initialization. It
    must return an iterable of complete rows, without acquiring hardware early.
    """

    factory: Callable[[], Iterable[RecordTuple]]
    columns: Sequence[str]
    file_path: str | Path


@dataclass(slots=True)
class MeasurementRecipe:
    """
    Declarative description of a measurement.

    `measure_mods`, `args`, `wrapper_lst`, and `compliance_lst` map directly to
    `MeasureManager.get_measure_dict`. The remaining fields describe the common
    execution behavior shared by top-level measurement methods.
    """

    measure_mods: tuple[str, ...]
    args: tuple[Any, ...] = ()
    wrapper_lst: list[Any] = field(default_factory=list)
    compliance_lst: list[Any] = field(default_factory=list)
    measure_kwargs: dict[str, Any] = field(default_factory=dict)
    step_time: float = 0
    extra_record_columns: Sequence[str] = ()
    extra_record_getters: Sequence[RecordGetter] = ()
    plot: Optional[PlotRecipe] = None
    on_record: Optional[RecordHook] = None
    on_measure: Optional[MeasureHook] = None
    after_prepare: Optional[PrepareHook] = None
    shutdown: Sequence[ShutdownTarget] = ()
    record_source: RecordSource | None = None


class MeasureFlow(LegacyMeasureFlow):
    """
    User-facing measurement flow.

    Existing legacy measurement methods are inherited from `measure_flow_old.py`
    with `_legacy` suffixes. New flows should be implemented as thin public
    methods that build a `MeasurementRecipe` and pass it to `run_recipe`.
    """

    def scan_recipe(
        self, plan: ScanPlan, recipe: MeasurementRecipe, *,
        reading_columns: Sequence[str], wait: Callable[[], None] | None = None,
    ) -> MeasurementRecipe:
        """Adapt two manual I/V sweeps and sense modules to an indexed scan.

        Build the ordinary recipe with MeasureModules/RecipeBuilder, ordering
        the sources like plan.source_names, then pass this result to run_recipe.
        This adapter owns sweep_tables, if_combine_gen, with_timer and
        manual_record_columns. All instrument, naming, plot and shutdown options
        stay on the ordinary recipe. AC senses require four reading column names.

        wait runs after both source generators advance and before any sensing;
        source_wait still controls each source's own delay. after_prepare sees
        the recombined rows and their corrected sweep-column indices.
        """
        if recipe.record_source is not None:
            logger.raise_error("scan recipes require measurement modules", ValueError, include_traceback=False)
        sources, senses, externals = self.extract_info_mods(recipe.measure_mods, *recipe.args)
        if len(sources) != 2 or externals or not senses or any(
            not any(config["sweep_fix"] == "sweep" and config["mode"] == "manual"
                    and config["ac_dc"] in ("ac", "dc") for config in source.values())
            for source in sources
        ):
            logger.raise_error("scan recipes require two manual I/V sweeps and sense modules",
                               ValueError, include_traceback=False)
        if len(reading_columns) != sum(4 if sense["ac_dc"] == "ac" else 1 for sense in senses):
            logger.raise_error("reading_columns must match the sense output width", ValueError, include_traceback=False)
        columns = (*plan.columns, "elapsed_s", *reading_columns)
        all_columns = (*columns, *recipe.extra_record_columns)
        if any(not isinstance(c, str) or not c for c in all_columns) or len(set(all_columns)) != len(all_columns):
            logger.raise_error("scan column names must be nonempty and unique", ValueError, include_traceback=False)

        def prepare_scan(mea_dict: dict[str, Any]) -> None:
            mea_dict["gen_lst"] = self.scan_records(plan, mea_dict["gen_lst"], wait=wait)
            # Plan indices precede the sources in scan rows, unlike ordinary rows.
            mea_dict["swp_idx"] = [columns.index(name) for name in plan.source_names]

        return replace(
            recipe,
            measure_kwargs={
                **recipe.measure_kwargs,
                # get_measure_dict consumes the outer list; a tuple gives each
                # preparation a fresh list so this recipe can be run again.
                "sweep_tables": tuple(plan.sweep_tables),
                "if_combine_gen": False, "with_timer": False,
                "manual_record_columns": list(columns),
            },
            after_prepare=add_after(prepare_scan, recipe.after_prepare or noop),
        )

    def scan_records(
        self, plan: ScanPlan, generators: Sequence[Iterator[Any]], *,
        wait: Callable[[], None] | None = None,
    ) -> Iterator[RecordTuple]:
        """Compose get_measure_dict's split generators; normally use scan_recipe.

        The first two generators are manual sources using plan.sweep_tables,
        followed by sense generators, with no timer. The shared combiner flattens
        multi-value readings (e.g. lock-in X/Y/R/phase) just as in ordinary runs.
        """
        source_rows = combined_generator_list(list(generators[:2]))
        read_rows = combined_generator_list(list(generators[2:]))
        with ExitStack() as cleanup:
            for generator in (*generators, source_rows, read_rows):
                if hasattr(generator, "close"):
                    cleanup.callback(generator.close)
            start = time.monotonic()
            for values in plan.points.itertuples(index=False, name=None):
                try:
                    outputs = next(source_rows)
                    if outputs != list(values[-2:]):
                        logger.raise_error("source generators do not match the scan plan",
                                           ValueError, include_traceback=False)
                    if wait is not None:
                        wait()
                    readings = next(read_rows)
                except StopIteration:
                    # A scan has a known point count; early exhaustion is a
                    # failed run, not successful completion of a partial scan.
                    logger.raise_error(f"scan generator ended before point {values[0]}",
                                       RuntimeError, include_traceback=False)
                if not all(math.isfinite(v) for v in readings):
                    logger.raise_error("scan readings must be finite", ValueError, include_traceback=False)
                yield (*values, time.monotonic() - start, *readings)

    def pulse_records(self, steps: Iterable[PulseStep], *, apply: Callable[[PulseStep], None],
                      restore_read: Callable[[], None], read: Callable[[], float],
                      read_bias: float, window: ReadWindow,
                      before_read: Callable[[], None] | None = None):
        """Yield PULSE_COLUMNS rows, preserving repeated amplitudes as distinct events.

        apply configures the write output; restore_read returns all affected outputs
        to their read state, including after an apply failure. before_read can wait
        for gate leakage/temperature after restoration. Configure recipe.shutdown
        separately for final output-off behavior. No R is inferred from current.
        time_since_pulse_s is measured since the restore action completed.
        """
        if not math.isfinite(read_bias):
            logger.raise_error("read_bias must be finite", ValueError, include_traceback=False)
        start = time.monotonic()  # Sequence start, including all write and read actions
        for pulse_id, step in enumerate(steps):  # Unique event index, even for repeated amplitudes
            try:
                apply(step)
                time.sleep(step.width_s)
            finally:
                restore_read()
            pulse_end = time.monotonic()  # Time when restoration to the read state completed
            if before_read is not None:
                before_read()
            for sample_id, _, value in self.read_window(read, window):  # Per-pulse index and measured current
                now = time.monotonic()  # Timestamp used for both output time columns
                yield (step.cycle_id, pulse_id, sample_id, "read", step.value,
                       step.width_s, read_bias, now - start, now - pulse_end, value)

    def prepare_recipe(self, recipe: MeasurementRecipe) -> dict[str, Any]:
        """Create the measure dictionary for a recipe."""
        if recipe.record_source is not None:
            source = recipe.record_source
            if recipe.measure_mods or recipe.args or recipe.wrapper_lst or recipe.compliance_lst or recipe.measure_kwargs:
                logger.raise_error("record_source cannot be combined with measurement modules or options", ValueError, include_traceback=False)
            if len(recipe.extra_record_columns) != len(recipe.extra_record_getters):
                logger.raise_error("record_columns and record_getters must have the same length", ValueError, include_traceback=False)
            columns = [*source.columns, *recipe.extra_record_columns]
            if not source.columns or any(not isinstance(c, str) or not c for c in columns) or len(set(columns)) != len(columns):
                logger.raise_error("record columns must be nonempty, unique strings", ValueError, include_traceback=False)
            if not callable(source.factory):
                logger.raise_error("record source factory must be callable", TypeError, include_traceback=False)
            if getattr(self, "_csv_fast_writer", None) is not None:
                logger.raise_error("finish the active recording before starting a custom source", RuntimeError, include_traceback=False)
            path, record_num, plot_path = self.record_init(
                (), manual_columns=columns, file_path=source.file_path,
                with_timer=False, overwrite=False, numeric_format=".12f",
            )
            return dict(file_path=path, plot_record_path=plot_path,
                        record_num=record_num, vary_mod=[], gen_lst=source)
        measure_kwargs = dict(recipe.measure_kwargs)
        self._prepare_record_getters(recipe, measure_kwargs)
        return self.get_measure_dict(
            recipe.measure_mods,
            *recipe.args,
            wrapper_lst=recipe.wrapper_lst,
            compliance_lst=recipe.compliance_lst,
            **measure_kwargs,
        )

    def run_recipe(self, recipe: MeasurementRecipe) -> dict[str, Any]:
        """
        Execute a measurement recipe end-to-end.

        The runner owns the repeated flow mechanics: prepare generators, record each row, update optional live plots, stop plot saving, and shut down requested source outputs.

        Hooks are used for do mea_dict or record_tuple treatment, for recipe treatment, place them in this function directly

        Return:
            Just for record, irrelevant to actual running
        """
        # A rejected custom run must not close or shut down an existing run.
        if recipe.record_source is not None and getattr(self, "_csv_fast_writer", None) is not None:
            logger.raise_error("finish the active recording before starting a custom source", RuntimeError, include_traceback=False)
        with ExitStack() as cleanup:
            for target in reversed(recipe.shutdown):
                cleanup.callback(self._shutdown_recipe_targets, (target,))
            cleanup.callback(self._finalize_recipe_record)
            # set up measurement configuration
            mea_dict = self.prepare_recipe(recipe)

            # HOOK: hook modification
            after_prepare_hook: PrepareHook = add_after(recipe.after_prepare, MeasureFlow.basic_info_meadict)
            on_measure_hook: MeasureHook = recipe.on_measure or noop
            on_record_hook: RecordHook = recipe.on_record or noop

            # print infos and register on_record hook for varying action
            if mea_dict["vary_mod"]:
                vary_infos = MeasureFlow._extract_vary(mea_dict)
                logger.validate("vary_loop" in recipe.measure_kwargs, "vary_loop setting not found")
                logger.info(f"bound: {vary_infos[-1]}, loop: {recipe.measure_kwargs['vary_loop']}")
                vary_flag = False   
                def start_varys(*args, **kwargs):
                    # only support one-start vary(all modules start at same time)
                    nonlocal vary_flag
                    if not vary_flag:
                        for funci in vary_infos[0]:
                            funci()
                        vary_flag = True
                on_record_hook: RecordHook = add_after(on_record_hook, start_varys)

            # execute after_prepare hook
            after_prepare_hook(mea_dict)

            # start plotting and periodic plot saving
            plotobj = self._init_recipe_plot(recipe.plot, mea_dict)
            if plotobj is not None:
                cleanup.callback(self._stop_recipe_plot, plotobj)
                if recipe.plot.flush is not None:
                    # ExitStack runs callbacks in reverse: flush the final grid
                    # before stopping the saver, which captures the final figure.
                    cleanup.callback(recipe.plot.flush, plotobj)
            rows = mea_dict["gen_lst"]
            if isinstance(rows, RecordSource):
                rows = rows.factory()
            rows = iter(rows)
            if hasattr(rows, "close"):
                cleanup.callback(rows.close)
            mea_iter = self._iter_records(rows, on_measure_hook)
            while True:
                t0 = time.perf_counter()
                try:
                    record_tuple = next(mea_iter)
                except StopIteration:
                    break
                record_tuple = self._with_record_getters(record_tuple, recipe)
                self.record_update(
                    mea_dict["file_path"],
                    mea_dict["record_num"],
                    record_tuple,
                )
                on_record_hook(record_tuple)
                if (
                    plotobj is not None
                    and recipe.plot is not None
                    and recipe.plot.update is not None
                ):
                    plot_records = self._with_plot_getters(record_tuple, recipe.plot)
                    recipe.plot.update(plotobj, plot_records)
                elapsed = time.perf_counter() - t0
                if recipe.step_time > 0:
                    time.sleep(max(0, recipe.step_time - elapsed))

        return mea_dict

    def _finalize_recipe_record(self) -> None:
        if getattr(self, "_csv_fast_writer", None) is not None:
            self.record_finalize()

    def _stop_recipe_plot(self, plotobj) -> None:
        try:
            plotobj.stop_saving()
        finally:
            self._active_plotobj = None

    _run_recipe = run_recipe

    @staticmethod
    def basic_info_meadict(mea_dict: dict)->None:
        logger.info("filepath: %s", mea_dict["file_path"])
        logger.info("no of columns(with time column): %d", mea_dict["record_num"])
        if mea_dict["vary_mod"]:
            logger.info(f"vary module: {mea_dict['vary_mod']}")

    @staticmethod
    def _iter_records(
        gen_lst: Iterable[RecordTuple], on_measure: MeasureHook = noop 
    ) -> Iterable[RecordTuple]:
        for record_tuple in gen_lst:
            # here can add steps for execution during measurement
            on_measure(record_tuple)
            yield record_tuple

    @staticmethod
    def _prepare_record_getters(
        recipe: MeasurementRecipe, measure_kwargs: dict[str, Any]
    ) -> None:
        """add extra records from recipe to measure_kwargs"""
        logger.validate(
            len(recipe.extra_record_columns) == len(recipe.extra_record_getters),
            "record_columns and record_getters must have the same length",
        )
        if not recipe.extra_record_getters:
            return
        logger.validate(
            "extra_record_columns" not in measure_kwargs,
            "extra_record_columns is managed by MeasurementRecipe.extra_record_columns, not manually",
        )
        measure_kwargs["extra_record_columns"] = tuple(recipe.extra_record_columns)

    @staticmethod
    def _with_record_getters(
        record_tuple: RecordTuple, recipe: MeasurementRecipe
    ) -> RecordTuple:
        if not recipe.extra_record_getters:
            return record_tuple
        return (*record_tuple, *(getter() for getter in recipe.extra_record_getters))

    @staticmethod
    def _with_plot_getters(
        record_tuple: RecordTuple, plotrec: PlotRecipe
    ) -> RecordTuple:
        if not plotrec.extra_getters:
            return record_tuple
        return (*record_tuple, *(getter() for getter in plotrec.extra_getters))

    def _init_recipe_plot(
        self, plotrec: Optional[PlotRecipe], mea_dict: dict[str, Any]
    ) -> Optional[DataManipulator]:
        if plotrec is None or not plotrec.enabled:
            return None

        plotobj = plotrec.plotobj if plotrec.plotobj is not None else DataManipulator(5)
        if plotrec.inline_jupyter is not None:
            plotrec.init_kwargs.setdefault("inline_jupyter", plotrec.inline_jupyter)
        with ExitStack() as cleanup:
            cleanup.callback(self._stop_recipe_plot, plotobj)
            if plotrec.init_args or plotrec.init_kwargs:
                plotobj.live_plot_init(*plotrec.init_args, **plotrec.init_kwargs)
            if plotrec.saving_interval is not None:
                plotobj.start_saving(mea_dict["plot_record_path"], plotrec.saving_interval)
            self._active_plotobj = plotobj
            cleanup.pop_all()
        return plotobj

    @staticmethod
    def _shutdown_recipe_targets(targets: Sequence[ShutdownTarget]) -> None:
        for target in targets:
            if callable(target) and not hasattr(target, "output_switch"):
                target()
            else:
                target.output_switch("off")

    
    @staticmethod
    def _extract_vary(
        mea_dict: dict
    ) -> tuple[list[Callable], list[Callable], list[Callable], list[Callable]]:
        """
        extract the vary functions, current value getters and set value getters for using in varying cases

        Currently the vary number is limited to 1
        """

        vary_lst = []
        curr_val_lst = []
        set_val_lst = []
        vary_bound_lst = []
        for i in mea_dict["vary_mod"]:
            match i:
                case "T":
                    vary_lst.append(mea_dict["tmp_vary"][0])
                    curr_val_lst.append(mea_dict["tmp_vary"][1])
                    set_val_lst.append(mea_dict["tmp_vary"][2])
                    vary_bound_lst.append(mea_dict["tmp_vary"][3])
                case "B":
                    vary_lst.append(mea_dict["mag_vary"][0])
                    curr_val_lst.append(mea_dict["mag_vary"][1])
                    set_val_lst.append(mea_dict["mag_vary"][2])
                    vary_bound_lst.append(mea_dict["mag_vary"][3])
                case "Theta" | "a" | "angle":
                    vary_lst.append(mea_dict["angle_vary"][0])
                    curr_val_lst.append(mea_dict["angle_vary"][1])
                    set_val_lst.append(mea_dict["angle_vary"][2])
                    vary_bound_lst.append(mea_dict["angle_vary"][3])

        logger.validate(
            min(len(vary_lst), len(curr_val_lst), len(set_val_lst), len(vary_bound_lst))
            == 1,
            "only one varying parameter is allowed",
        )

        return vary_lst, curr_val_lst, set_val_lst, vary_bound_lst
