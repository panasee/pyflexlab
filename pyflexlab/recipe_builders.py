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


def _split_source_sense_meter(meter: Meter | list[Meter]) -> list[Meter]:
    if isinstance(meter, list):
        logger.validate(len(meter) == 2, "meter must be a list of two meters")
        return meter
    return [meter, meter]


def _build_range_prepare_hook(
    src_sens_lst: list[Meter],
    *,
    sense_range: float | None,
    source_range: float | None,
):
    if sense_range is None and source_range is None:
        return None

    def after_prepare(mea_dict: dict) -> None:
        if sense_range is not None:
            src_sens_lst[1].sense_range_volt = sense_range
            logger.info("sense range: %s", sense_range)
        if source_range is not None:
            src_sens_lst[0].source_range = source_range
            logger.info("source range: %s", source_range)

    return after_prepare


def fixed_current_source(
    value: float,
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


def fixed_voltage_source(
    value: float,
    *,
    high: int | str,
    low: int | str,
    meter: Meter,
    compliance: float | str,
) -> RecipeModule:
    return RecipeModule(
        module_id="source.fixed_voltage",
        category="source",
        measure_mod="V_source_fixed_dc",
        args=(value, high, low, ""),
        wrapper=meter,
        compliance=compliance,
    )


def sweep_voltage_source(
    max_value: float,
    step_value: float,
    *,
    high: int | str,
    low: int | str,
    sweepmode: str,
    meter: Meter,
    compliance: float | str,
    freq: float | None = None,
) -> RecipeModule:
    if freq is None:
        return RecipeModule(
            module_id="source.sweep_voltage",
            category="source",
            measure_mod="V_source_sweep_dc",
            args=(max_value, step_value, high, low, sweepmode, ""),
            wrapper=meter,
            compliance=compliance,
        )
    return RecipeModule(
        module_id="source.sweep_voltage",
        category="source",
        measure_mod="V_source_sweep_ac",
        args=(max_value, step_value, freq, high, low, sweepmode, ""),
        wrapper=meter,
        compliance=compliance,
    )


def voltage_sense(
    *,
    high: int | str,
    low: int | str,
    comment: str = "",
    meter: Meter,
    freq: float | None = None,
) -> RecipeModule:
    return RecipeModule(
        module_id="sense.voltage",
        category="sense",
        measure_mod="V_sense_dc" if freq is None else "V_sense_ac",
        args=(high, low, comment),
        wrapper=meter,
    )


def current_sense(
    *,
    high: int | str,
    low: int | str,
    comment: str = "",
    meter: Meter,
    freq: float | None = None,
) -> RecipeModule:
    return RecipeModule(
        module_id="sense.current",
        category="sense",
        measure_mod="I_sense_dc" if freq is None else "I_sense_ac",
        args=(high, low, comment),
        wrapper=meter,
    )


def vary_magnetic_field(*, start: float, stop: float) -> RecipeModule:
    return RecipeModule(
        module_id="external.vary_magnetic_field",
        category="external",
        measure_mod="B_vary",
        args=(start, stop),
    )


def fixed_magnetic_field(value: float) -> RecipeModule:
    return RecipeModule(
        module_id="external.fixed_magnetic_field",
        category="external",
        measure_mod="B_fixed",
        args=(value,),
    )


def fixed_temperature(value: float) -> RecipeModule:
    return RecipeModule(
        module_id="external.fixed_temperature",
        category="external",
        measure_mod="T_fixed",
        args=(value,),
    )


def _resolve_sense_terminals(
    *,
    source_high: int | str,
    source_low: int | str,
    sense_high: int | str | None,
    sense_low: int | str | None,
) -> tuple[int | str, int | str]:
    logger.validate(
        (sense_high is None) == (sense_low is None),
        "sense_high and sense_low must be set together",
    )
    if sense_high is None:
        return source_high, source_low
    logger.info("four wire terminals detected, please check meter setting")
    return sense_high, sense_low


def _vi_curve_dc_update(plotobj: DataManipulator, records: RecordTuple) -> None:
    logger.validate(len(records) >= 3, "V-I dc plot needs at least 3 record columns")
    plotobj.live_plot_update(0, 0, 0, records[1], records[2], incremental=True)


def _vi_curve_ac_update(plotobj: DataManipulator, records: RecordTuple) -> None:
    logger.validate(len(records) >= 4, "V-I ac plot needs at least 4 record columns")
    plotobj.live_plot_update(
        [0, 0],
        [0, 0],
        [0, 1],
        [records[1], records[1]],
        [records[2], records[3]],
        incremental=True,
    )


def _rh_loop_dc_update(plotobj: DataManipulator, records: RecordTuple) -> None:
    logger.validate(
        len(records) >= 7, "R-H loop dc plot needs at least 7 record columns"
    )
    plotobj.live_plot_update(
        [0, 0],
        [0, 1],
        [0, 0],
        [records[5], records[5]],
        [records[3], records[6]],
        incremental=True,
    )
    plotobj.live_plot_update(0, 2, 0, records[0], records[5], incremental=True)


def _rh_loop_ac_update(plotobj: DataManipulator, records: RecordTuple) -> None:
    logger.validate(
        len(records) >= 10, "R-H loop ac plot needs at least 10 record columns"
    )
    plotobj.live_plot_update(
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [records[8], records[8], records[8]],
        [records[3], records[4], records[9]],
        incremental=True,
    )
    plotobj.live_plot_update(0, 2, 0, records[0], records[8], incremental=True)


def _build_vi_curve_plot(
    *,
    freq: Optional[float],
    saving_interval: float,
    plotobj: Optional[DataManipulator],
    use_dash: bool,
) -> PlotRecipe:
    return PlotRecipe(
        plotobj=plotobj,
        init_args=(1, 1, 1 if freq is None else 2),
        init_kwargs={
            "titles": [[r"$V-I Curve$"]],
            "axes_labels": [[[r"$V$", r"$I$"]]],
            "line_labels": [[["V-I"]]]
            if freq is None
            else [[["V-I-x", "V-I-y"]]],
        },
        update=_vi_curve_dc_update if freq is None else _vi_curve_ac_update,
        saving_interval=saving_interval,
        inline_jupyter=not use_dash,
    )


def _build_rh_loop_plot(
    *,
    freq: Optional[float],
    saving_interval: float,
    plotobj: Optional[DataManipulator],
    use_dash: bool,
) -> PlotRecipe:
    return PlotRecipe(
        plotobj=plotobj,
        init_args=(1, 3, 1 if freq is None else 2),
        init_kwargs={"titles": [["B I", "B T", "t B"]]},
        update=_rh_loop_dc_update if freq is None else _rh_loop_ac_update,
        saving_interval=saving_interval,
        inline_jupyter=not use_dash,
    )


def vi_curve_plot(
    *,
    freq: Optional[float] = None,
    saving_interval: float = 7,
    plotobj: Optional[DataManipulator] = None,
    use_dash: bool = False,
) -> PlotRecipe:
    return _build_vi_curve_plot(
        freq=freq,
        saving_interval=saving_interval,
        plotobj=plotobj,
        use_dash=use_dash,
    )


def rh_loop_plot(
    *,
    freq: Optional[float] = None,
    saving_interval: float = 7,
    plotobj: Optional[DataManipulator] = None,
    use_dash: bool = False,
) -> PlotRecipe:
    return _build_rh_loop_plot(
        freq=freq,
        saving_interval=saving_interval,
        plotobj=plotobj,
        use_dash=use_dash,
    )


def build_vi_curve_recipe(
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
) -> MeasurementRecipe:
    """Build the recipe equivalent of ``measure_Vswp_I_vicurve_legacy``."""

    src_sens_lst = _split_source_sense_meter(meter)
    return assemble_recipe(
        sweep_voltage_source(
            vmax,
            vstep,
            high=high,
            low=low,
            sweepmode=swpmode,
            meter=src_sens_lst[0],
            compliance=compliance,
            freq=freq,
        ),
        current_sense(
            high=high,
            low=low,
            meter=src_sens_lst[1],
            freq=freq,
        ),
        options=RecipeOptions(
            special_name=folder_name,
            measure_nickname="vi-curve-dc" if freq is None else "vi-curve-ac",
            source_wait=source_wait if freq is None else None,
        ),
        step_time=step_time,
        plot=vi_curve_plot(
            freq=freq,
            saving_interval=saving_interval,
            plotobj=plotobj,
            use_dash=use_dash,
        )
        if if_plot
        else None,
        after_prepare=_build_range_prepare_hook(
            src_sens_lst,
            sense_range=sense_range,
            source_range=source_range,
        ),
        shutdown=(src_sens_lst[0],),
    )


def build_iv_vi_bvaryt_rhloop_recipe(
    *,
    ids: float,
    ds_high: int | str,
    ds_low: int | str,
    ds_meter: Meter | list[Meter],
    ds_compliance: float | str,
    vg: float,
    vg_high: int | str,
    vg_meter: Meter,
    vg_compliance: float | str,
    field_start: float,
    field_end: float,
    temperature: float,
    sense_high: int | str | None = None,
    sense_low: int | str | None = None,
    freq: Optional[float] = None,
    folder_name: str = "",
    measure_nickname: str | None = None,
    step_time: float = 0.3,
    wait_before_vary: float = 5,
    vary_loop: bool = True,
    if_plot: bool = True,
    saving_interval: float = 7,
    plotobj: Optional[DataManipulator] = None,
    source_wait: float | None = None,
    use_dash: bool = False,
    after_prepare: Optional[PrepareHook] = None,
    on_measure: Optional[MeasureHook] = None,
    on_record: Optional[RecordHook] = None,
) -> MeasurementRecipe:
    """Build the recipe equivalent of ``measure_IV_VI_BvaryT_rhloop_legacy``."""

    ds_src_sens_lst = _split_source_sense_meter(ds_meter)
    resolved_sense_high, resolved_sense_low = _resolve_sense_terminals(
        source_high=ds_high,
        source_low=ds_low,
        sense_high=sense_high,
        sense_low=sense_low,
    )

    return assemble_recipe(
        fixed_current_source(
            ids,
            high=ds_high,
            low=ds_low,
            meter=ds_src_sens_lst[0],
            compliance=ds_compliance,
            freq=freq,
        ),
        fixed_voltage_source(
            vg,
            high=vg_high,
            low=0,
            meter=vg_meter,
            compliance=vg_compliance,
        ),
        voltage_sense(
            high=resolved_sense_high,
            low=resolved_sense_low,
            meter=ds_src_sens_lst[1],
            freq=freq,
        ),
        current_sense(high=vg_high, low=0, meter=vg_meter),
        vary_magnetic_field(start=field_start, stop=field_end),
        fixed_temperature(temperature),
        options=RecipeOptions(
            if_combine_gen=True,
            special_name=folder_name,
            measure_nickname=measure_nickname
            or ("rh-loop-dc" if freq is None else "rh-loop-ac"),
            vary_loop=vary_loop,
            wait_before_vary=wait_before_vary,
            source_wait=source_wait,
        ),
        step_time=step_time,
        plot=rh_loop_plot(
            freq=freq,
            saving_interval=saving_interval,
            plotobj=plotobj,
            use_dash=use_dash,
        )
        if if_plot
        else None,
        after_prepare=after_prepare,
        on_measure=on_measure,
        on_record=on_record,
    )
