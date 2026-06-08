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
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence
from functools import wraps

from pyomnix.data_process import DataManipulator
from pyomnix.omnix_logger import get_logger

from pyflexlab.equip_wrapper import Meter, SourceMeter
from pyflexlab.measure_flow_old import MeasureFlow as LegacyMeasureFlow

logger = get_logger(__name__)

RecordTuple = tuple[Any, ...]  # actual measured values tuple from the whole generator
PlotUpdate = Callable[[DataManipulator, RecordTuple], None]
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
    saving_interval: float = 7
    inline_jupyter: Optional[bool] = None


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
    plot: Optional[PlotRecipe] = None
    on_record: Optional[RecordHook] = None
    on_measure: Optional[MeasureHook] = None
    after_prepare: Optional[PrepareHook] = None
    shutdown: Sequence[ShutdownTarget] = ()


class MeasureFlow(LegacyMeasureFlow):
    """
    User-facing measurement flow.

    Existing legacy measurement methods are inherited from `measure_flow_old.py`
    with `_legacy` suffixes. New flows should be implemented as thin public
    methods that build a `MeasurementRecipe` and pass it to `run_recipe`.
    """

    def prepare_recipe(self, recipe: MeasurementRecipe) -> dict[str, Any]:
        """Create the measure dictionary for a recipe."""
        measure_kwargs = dict(recipe.measure_kwargs)
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
        # set up measurement configuration
        mea_dict = self.prepare_recipe(recipe)

        # HOOK: hook modification
        after_prepare_hook: PrepareHook = add_after(recipe.after_prepare, MeasureFlow.basic_info_meadict)
        on_measure_hook: MeasureHook = recipe.on_measure or noop

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
            on_record_hook: RecordHook = add_after(recipe.on_record, start_varys)

        # execute after_prepare hook
        after_prepare_hook(mea_dict)

        # start plotting and periodic plot saving
        plotobj = self._init_recipe_plot(recipe.plot, mea_dict)
        try:
            mea_iter = self._iter_records(mea_dict["gen_lst"], on_measure_hook)
            while True:
                t0 = time.perf_counter()
                try:
                    record_tuple = next(mea_iter)
                except StopIteration:
                    break
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
                    recipe.plot.update(plotobj, record_tuple)
                elapsed = time.perf_counter() - t0
                if recipe.step_time > 0:
                    time.sleep(max(0, recipe.step_time - elapsed))
        finally:
            if plotobj is not None:
                plotobj.stop_saving()
                self._active_plotobj = None
            self._shutdown_recipe_targets(recipe.shutdown)

        return mea_dict

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

    def _init_recipe_plot(
        self, plotrec: Optional[PlotRecipe], mea_dict: dict[str, Any]
    ) -> Optional[DataManipulator]:
        if plotrec is None or not plotrec.enabled:
            return None

        plotobj = plotrec.plotobj if plotrec.plotobj is not None else DataManipulator(5)
        if plotrec.inline_jupyter is not None:
            plotrec.init_kwargs.setdefault("inline_jupyter", plotrec.inline_jupyter)
        if plotrec.init_args or plotrec.init_kwargs:
            plotobj.live_plot_init(*plotrec.init_args, **plotrec.init_kwargs)
        plotobj.start_saving(mea_dict["plot_record_path"], plotrec.saving_interval)
        self._active_plotobj = plotobj
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
                case "Theta":
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

    # template for using recipe_builder
    def measure_Vswp_I_vicurve(
        self,
        *,
        vmax: float,
        vstep: float,
        freq: Optional[float] = None,
        high: int | str,
        low: int | str,
        swpmode: str,
        meter: Meter | list[Meter],
        compliance: float,
        folder_name: str = "",
        step_time: float = 0.5,
        if_plot: bool = True,
        saving_interval: float = 7,
        plotobj: Optional[DataManipulator] = None,
        source_wait: float = 0.1,
        use_dash: bool = False,
        sense_range: float | None = None,
        source_range: float | None = None,
    ) -> dict[str, Any]:
        """Measure a V-I curve through the reusable recipe runner."""
        from .recipe_builders import build_vi_curve_recipe

        recipe = build_vi_curve_recipe(
            vmax=vmax,
            vstep=vstep,
            freq=freq,
            high=high,
            low=low,
            swpmode=swpmode,
            meter=meter,
            compliance=compliance,
            folder_name=folder_name,
            step_time=step_time,
            if_plot=if_plot,
            saving_interval=saving_interval,
            plotobj=plotobj,
            source_wait=source_wait,
            use_dash=use_dash,
            sense_range=sense_range,
            source_range=source_range,
        )
        return self.run_recipe(recipe)
