"""Reusable recipe builders for common measurement workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence

from pyomnix.data_process import DataManipulator
from pyomnix.omnix_logger import get_logger
from pyflexlab.equip_wrapper import Meter
from pyflexlab.measure_flow import (
    MeasurementRecipe,
    MeasureHook,
    PlotRecipe,
    PrepareHook,
    RecordHook,
    RecordTuple,
)

logger = get_logger(__name__)

RecipeModuleCategory = Literal["source", "sense", "external"]


@dataclass(frozen=True, slots=True)
class RecipeModule:
    """One named fragment that maps to one MeasureManager measurement module."""

    module_id: str
    category: RecipeModuleCategory
    measure_mod: str
    args: tuple[Any, ...] = ()
    wrapper: Any = None
    compliance: Any = None


@dataclass(frozen=True, slots=True)
class PlotSeries:
    """One record-column mapping into one live-plot panel line."""
    row: int
    col: int
    line: int
    x_col: int
    y_col: int
    x_label: str = ""
    y_label: str = ""
    line_label: str | None = None


@dataclass(frozen=True, slots=True)
class RecipeOptions:
    """Common get_measure_dict options shared by assembled recipes."""

    special_name: str = ""
    measure_nickname: str | None = None
    if_combine_gen: bool | None = None
    with_timer: bool | None = None
    vary_loop: bool | None = None
    wait_before_vary: float | None = None
    source_wait: float | None = None
    appendix_str: str | None = None
    allow_large_jump: bool | None = None
    extra_measure_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_measure_kwargs(self) -> dict[str, Any]:
        """output dataclass to dict for calling"""
        kwargs = dict(self.extra_measure_kwargs)
        kwargs["special_name"] = self.special_name
        optional_values = {
            "measure_nickname": self.measure_nickname,
            "if_combine_gen": self.if_combine_gen,
            "with_timer": self.with_timer,
            "vary_loop": self.vary_loop,
            "wait_before_vary": self.wait_before_vary,
            "source_wait": self.source_wait,
            "appendix_str": self.appendix_str,
            "allow_large_jump": self.allow_large_jump,
        }
        for key, value in optional_values.items():
            if value is not None:
                kwargs[key] = value
        return kwargs


def _ordered_modules(
    modules: Sequence[RecipeModule],
) -> tuple[list[RecipeModule], list[RecipeModule], list[RecipeModule]]:
    sources = [module for module in modules if module.category == "source"]
    senses = [module for module in modules if module.category == "sense"]
    externals = [module for module in modules if module.category == "external"]
    logger.validate(
        len(sources) + len(senses) + len(externals) == len(modules),
        "unsupported recipe module category",
    )
    return sources, senses, externals


def assemble_recipe(
    *modules: RecipeModule,
    options: RecipeOptions | None = None,
    step_time: float = 0,
    plot: Optional[PlotRecipe] = None,
    after_prepare: Optional[PrepareHook] = None,
    on_measure: Optional[MeasureHook] = None,
    on_record: Optional[RecordHook] = None,
    shutdown: Sequence[Any] = (),
) -> MeasurementRecipe:
    """Assemble named source/sense/external fragments into a MeasurementRecipe."""

    sources, senses, externals = _ordered_modules(modules)
    ordered_modules = [*sources, *senses, *externals]
    logger.validate(
        all(module.wrapper is not None for module in [*sources, *senses]),
        "source and sense recipe modules require wrapper meters",
    )
    return MeasurementRecipe(
        measure_mods=tuple(module.measure_mod for module in ordered_modules),
        args=tuple(arg for module in ordered_modules for arg in module.args),
        wrapper_lst=[module.wrapper for module in [*sources, *senses]],
        compliance_lst=[module.compliance for module in sources],
        measure_kwargs=(options or RecipeOptions()).to_measure_kwargs(),
        step_time=step_time,
        plot=plot,
        after_prepare=after_prepare,
        on_measure=on_measure,
        on_record=on_record,
        shutdown=shutdown,
    )


class RecipeBuilder:
    """Small mutable helper for interactive or GUI-style recipe assembly."""

    def __init__(self, *, options: RecipeOptions | None = None) -> None:
        self._modules: list[RecipeModule] = []
        self.options = options or RecipeOptions()

    def add(self, module: RecipeModule) -> "RecipeBuilder":
        self._modules.append(module)
        return self

    def build(
        self,
        *,
        step_time: float = 0,
        plot: Optional[PlotRecipe] = None,
        after_prepare: Optional[PrepareHook] = None,
        on_measure: Optional[MeasureHook] = None,
        on_record: Optional[RecordHook] = None,
        shutdown: Sequence[Any] = (),
    ) -> MeasurementRecipe:
        return assemble_recipe(
            *self._modules,
            options=self.options,
            step_time=step_time,
            plot=plot,
            after_prepare=after_prepare,
            on_measure=on_measure,
            on_record=on_record,
            shutdown=shutdown,
        )


# ==================measure modules==============
class MeasureModules:
    @staticmethod
    def fixed_current_source(
        value: float | str,
        *,
        high: int | str,
        low: int | str,
        meter: Meter,
        compliance: float | str,
        freq: float | None = None,
    ) -> RecipeModule:
        if freq is None:
            return RecipeModule(
                module_id="source.fixed_current",
                category="source",
                measure_mod="I_source_fixed_dc",
                args=(value, high, low),
                wrapper=meter,
                compliance=compliance,
            )
        return RecipeModule(
            module_id="source.fixed_current",
            category="source",
            measure_mod="I_source_fixed_ac",
            args=(value, freq, high, low),
            wrapper=meter,
            compliance=compliance,
        )


    @staticmethod
    def fixed_voltage_source(
        value: float | str,
        *,
        high: int | str,
        low: int | str,
        meter: Meter,
        compliance: float | str,
        freq: float | None = None,
    ) -> RecipeModule:
        if freq is None:
            return RecipeModule(
                module_id="source.fixed_voltage",
                category="source",
                measure_mod="V_source_fixed_dc",
                args=(value, high, low),
                wrapper=meter,
                compliance=compliance,
            )
        return RecipeModule(
            module_id="source.fixed_voltage",
            category="source",
            measure_mod="V_source_fixed_ac",
            args=(value, freq, high, low),
            wrapper=meter,
            compliance=compliance,
        )


    @staticmethod
    def sweep_voltage_source(
        max_value: float | str,
        step_value: float | str,
        *,
        high: int | str,
        low: int | str,
        sweepmode: Literal[
            "0-max-0", "0--max-max-0", "0-max--max-max-0", "0-max", "manual"
        ],
        meter: Meter,
        compliance: float | str,
        freq: float | None = None,
    ) -> RecipeModule:
        if freq is None:
            return RecipeModule(
                module_id="source.sweep_voltage",
                category="source",
                measure_mod="V_source_sweep_dc",
                args=(max_value, step_value, high, low, sweepmode),
                wrapper=meter,
                compliance=compliance,
            )
        return RecipeModule(
            module_id="source.sweep_voltage",
            category="source",
            measure_mod="V_source_sweep_ac",
            args=(max_value, step_value, freq, high, low, sweepmode),
            wrapper=meter,
            compliance=compliance,
        )


    @staticmethod
    def sweep_current_source(
        max_value: float | str,
        step_value: float | str,
        *,
        high: int | str,
        low: int | str,
        sweepmode: Literal[
            "0-max-0", "0--max-max-0", "0-max--max-max-0", "0-max", "manual"
        ],
        meter: Meter,
        compliance: float | str,
        freq: float | None = None,
    ) -> RecipeModule:
        if freq is None:
            return RecipeModule(
                module_id="source.sweep_current",
                category="source",
                measure_mod="V_source_sweep_dc",
                args=(max_value, step_value, high, low, sweepmode),
                wrapper=meter,
                compliance=compliance,
            )
        return RecipeModule(
            module_id="source.sweep_current",
            category="source",
            measure_mod="V_source_sweep_ac",
            args=(max_value, step_value, freq, high, low, sweepmode),
            wrapper=meter,
            compliance=compliance,
        )


    @staticmethod
    def voltage_sense(
        *,
        high: int | str,
        low: int | str,
        comment: str = "",
        meter: Meter,
        ac_dc: Literal["ac", "dc"],
    ) -> RecipeModule:
        return RecipeModule(
            module_id="sense.voltage",
            category="sense",
            measure_mod=f"V_sense_{ac_dc}",
            args=(comment, high, low),
            wrapper=meter,
        )

    @staticmethod
    def current_sense(
        *,
        high: int | str,
        low: int | str,
        comment: str = "",
        meter: Meter,
        ac_dc: Literal["ac", "dc"],
    ) -> RecipeModule:
        return RecipeModule(
            module_id="sense.current",
            category="sense",
            measure_mod=f"I_sense_{ac_dc}",
            args=(comment, high, low),
            wrapper=meter,
        )


    @staticmethod
    def vary_magnetic_field(*, start: float | str, stop: float | str) -> RecipeModule:
        return RecipeModule(
            module_id="external.vary_magnetic_field",
            category="external",
            measure_mod="B_vary",
            args=(start, stop),
        )


    @staticmethod
    def sweep_magnetic_field(
        *, start: float | str, stop: float | str, step: float | str, 
        sweepmode: Literal["0-max-0", "0--max-max-0", "min-max", "manual"],
    ) -> RecipeModule:
        return RecipeModule(
            module_id="external.sweep_magnetic_field",
            category="external",
            measure_mod="B_sweep",
            args=(start, stop, step, sweepmode),
        )


    @staticmethod
    def fixed_magnetic_field(value: float | str) -> RecipeModule:
        return RecipeModule(
            module_id="external.fixed_magnetic_field",
            category="external",
            measure_mod="B_fixed",
            args=(value,),
        )

    @staticmethod
    def vary_temperature(*, start: float | str, stop: float | str) -> RecipeModule:
        return RecipeModule(
            module_id="external.vary_temperature",
            category="external",
            measure_mod="T_vary",
            args=(start, stop),
        )


    @staticmethod
    def sweep_temperature(
        *, start: float | str, stop: float | str, step: float | str, 
        sweepmode: Literal["0-max-0", "0--max-max-0", "min-max", "manual"],
    ) -> RecipeModule:
        return RecipeModule(
            module_id="external.sweep_temperature",
            category="external",
            measure_mod="T_sweep",
            args=(start, stop, step, sweepmode),
        )


    @staticmethod
    def fixed_temperature(value: float) -> RecipeModule:
        return RecipeModule(
            module_id="external.fixed_temperature",
            category="external",
            measure_mod="T_fixed",
            args=(value,),
        )


# ===========plot modules=============
class PlotModules:
    @staticmethod
    def mapped_plot(
        *,
        init_args: tuple[Any, ...],
        series: Sequence[PlotSeries],
        titles: list[list[str]] | None = None,
        saving_interval: float = 7,
        plotobj: Optional[DataManipulator] = None,
        use_dash: bool = False,
        init_kwargs: dict[str, Any] | None = None,
        update_kwargs: dict[str, Any] | None = None,
    ) -> PlotRecipe:
        logger.validate(series, "mapped_plot requires at least one PlotSeries")
        plot_init_kwargs = dict(init_kwargs or {})
        plot_init_kwargs["axes_labels"] = PlotModules._build_axes_labels(
            init_args=init_args,
            series=series,
        )
        if titles is not None:
            plot_init_kwargs["titles"] = titles
        line_labels = PlotModules._build_line_labels(
            init_args=init_args,
            series=series,
        )
        if any(any(row) for row in line_labels):
            plot_init_kwargs["line_labels"] = line_labels

        return PlotRecipe(
            plotobj=plotobj,
            init_args=init_args,
            init_kwargs=plot_init_kwargs,
            update=PlotModules._build_update(series, update_kwargs=update_kwargs),
            saving_interval=saving_interval,
            inline_jupyter=not use_dash,
        )

    @staticmethod
    def _build_axes_labels(
        *,
        init_args: tuple[Any, ...],
        series: Sequence[PlotSeries],
    ) -> list[list[list[str]]]:
        rows = int(init_args[0]) if len(init_args) >= 1 else 1
        cols = int(init_args[1]) if len(init_args) >= 1 else 1
        axes_labels = [[["", ""] for _ in range(cols)] for _ in range(rows)]
        for item in series:
            axes_labels[item.row][item.col] = [item.x_label, item.y_label]
        return axes_labels

    @staticmethod
    def _build_line_labels(
        *,
        init_args: tuple[Any, ...],
        series: Sequence[PlotSeries],
    ) -> list[list[list[str]]]:
        rows = int(init_args[0]) if len(init_args) >= 1 else 1
        cols = int(init_args[1]) if len(init_args) >= 1 else 1
        line_labels = [[[] for _ in range(cols)] for _ in range(rows)]
        for item in series:
            if item.line_label is None:
                continue
            panel_labels = line_labels[item.row][item.col]
            while len(panel_labels) <= item.line:
                panel_labels.append("")
            panel_labels[item.line] = item.line_label
        return line_labels

    @staticmethod
    def _build_update(
        series: Sequence[PlotSeries],
        *,
        update_kwargs: dict[str, Any] | None = None,
    ):
        max_col = max(
            max(item.x_col, item.y_col)
            for item in series
        )
        plot_update_kwargs = dict(update_kwargs or {})
        logger.validate(
            "incremental" not in plot_update_kwargs,
            "mapped_plot update_kwargs must not set incremental",
        )

        def update(plotobj: DataManipulator, records: RecordTuple) -> None:
            logger.validate(
                len(records) > max_col,
                f"plot mapping needs record column {max_col}, got {len(records)} columns",
            )
            for item in series:
                plotobj.live_plot_update(
                    item.row,
                    item.col,
                    item.line,
                    records[item.x_col],
                    records[item.y_col],
                    incremental=True,
                    **plot_update_kwargs,
                )

        return update
