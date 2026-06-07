"""Reusable recipe builders for common measurement workflows."""

from __future__ import annotations

from typing import Optional

from pyomnix.data_process import DataManipulator
from pyomnix.omnix_logger import get_logger
from pyflexlab.equip_wrapper import Meter
from pyflexlab.measure_flow import MeasurementRecipe, PlotRecipe, RecordTuple

logger = get_logger(__name__)


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
    if freq is None:
        measure_mods = ("V_source_sweep_dc", "I_sense_dc")
        args = (vmax, vstep, high, low, swpmode, "", high, low)
        measure_kwargs = {
            "special_name": folder_name,
            "measure_nickname": "vi-curve-dc",
            "source_wait": source_wait,
        }
    else:
        measure_mods = ("V_source_sweep_ac", "I_sense_ac")
        args = (vmax, vstep, freq, high, low, swpmode, "", high, low)
        measure_kwargs = {
            "special_name": folder_name,
            "measure_nickname": "vi-curve-ac",
        }

    return MeasurementRecipe(
        measure_mods=measure_mods,
        args=args,
        wrapper_lst=src_sens_lst,
        compliance_lst=[compliance],
        measure_kwargs=measure_kwargs,
        step_time=step_time,
        plot=_build_vi_curve_plot(
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
