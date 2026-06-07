"""
High-level measurement flow interface.

The legacy method collection lives in measure_flow_old.py. This module keeps
those public methods available while adding a smaller recipe/runner layer for
new measurement flows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

from pyomnix.data_process import DataManipulator
from pyomnix.omnix_logger import get_logger

from .equip_wrapper import SourceMeter
from .measure_flow_old import MeasureFlow as LegacyMeasureFlow

logger = get_logger(__name__)

RecordTuple = tuple[Any, ...] # actual measured values tuple from the whole generator
PlotUpdate = Callable[[DataManipulator, RecordTuple], None]
RecordHook = Callable[[RecordTuple], None]
PrepareHook = Callable[[dict[str, Any]], None]
ShutdownTarget = SourceMeter | Callable[[], None]


@dataclass(slots=True)
class PlotRecipe:
    """Plot configuration for a recipe runner."""

    enabled: bool = True
    plotobj: Optional[DataManipulator] = None
    # args & kwargs are all directly fed into live_plot_init
    init_args: tuple[Any, ...] = ()
    init_kwargs: dict[str, Any] = field(default_factory=dict)

    update: Optional[PlotUpdate] = None
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

        The runner owns the repeated flow mechanics: prepare generators, record
        each row, update optional live plots, stop plot saving, and shut down
        requested source outputs.

        Return:
            Just for record, irrelevant to actual running
        """
        # set up measurement configuration
        mea_dict = self.prepare_recipe(recipe)
        ##TODO:: check##
        if recipe.after_prepare is not None:
            recipe.after_prepare(mea_dict)
        ##::TODO##

        # start plotting and periodic plot saving
        plotobj = self._init_recipe_plot(recipe.plot, mea_dict)
        logger.info("filepath: %s", mea_dict["file_path"])
        logger.info("no of columns(with time column): %d", mea_dict["record_num"])
        logger.info("vary modules: %s", mea_dict["vary_mod"])

        try:
            for record_tuple in self._iter_records(mea_dict["gen_lst"]):
                self.record_update(
                    mea_dict["file_path"],
                    mea_dict["record_num"],
                    record_tuple,
                )
                if recipe.on_record is not None:
                    recipe.on_record(record_tuple)
                if (
                    plotobj is not None
                    and recipe.plot is not None
                    and recipe.plot.update is not None
                ):
                    recipe.plot.update(plotobj, record_tuple)
                if recipe.step_time > 0:
                    time.sleep(recipe.step_time)
        finally:
            if plotobj is not None:
                plotobj.stop_saving()
                self._active_plotobj = None
            self._shutdown_recipe_targets(recipe.shutdown)

        return mea_dict

    _run_recipe = run_recipe

    @staticmethod
    def _iter_records(gen_lst: Iterable[RecordTuple]) -> Iterable[RecordTuple]:
        for record_tuple in gen_lst:
            # here can add steps for execution during measurement
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
