"""
The non-individual plot options have not been written yet
"""

import time
from typing import Sequence, Callable, Optional, Literal
import numpy as np
from prefect import task
from prefect.cache_policies import NO_CACHE
from .measure_manager import MeasureManager
from .equip_wrapper import Meter
from pyomnix.data_process import DataManipulator
from pyomnix.omnix_logger import get_logger

logger = get_logger(__name__)


class MeasureFlow(MeasureManager):
    """
    This class is a subclass of MeasureManager and is responsible for managing the measure-related folders and data
    """

    @task(name="vi-curve-simple", cache_policy=NO_CACHE)
    def measure_Vswp_I_vicurve_task(
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
        folder_name: str = None,
        step_time: float = 0.5,
        individual_plot: bool = True,
    ):
        self.measure_Vswp_I_vicurve(
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
            individual_plot=individual_plot,
        )

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
        individual_plot: bool = True,
        saving_interval: float = 7,
    ) -> None:
        """
        measure the V-I curve using ONE DC source meter, no other info (B, T, etc.). Use freq to indicate ac measurement

        Args:
            vmax: float, the maximum voltage
            vstep: float, the step voltage
            high: float, the high terminal of the voltage
            low: float, the low terminal of the voltage
            swpmode: str, the sweep mode
            meter: Meter | list[Meter], the meter used for both source and sense or two meters separately in a list
            compliance: float, the compliance
            freq: float, the frequency
            folder_name: str, the folder name
            step_time: float, the step time
            individual_plot: bool, the individual plot
            saving_interval: float, the saving interval in seconds
        """
        if isinstance(meter, list):
            logger.validate(len(meter) == 2, "meter must be a list of two meters")
            src_sens_lst = meter
        else:
            src_sens_lst = [meter, meter]
        if freq is None:
            mea_dict = self.get_measure_dict(
                ("V_source_sweep_dc", "I_sense"),
                vmax,
                vstep,
                high,
                low,
                swpmode,
                "",
                high,
                low,
                wrapper_lst=src_sens_lst,
                compliance_lst=[compliance],
                special_name=folder_name,
                measure_nickname="vi-curve-dc",
            )
        else:
            mea_dict = self.get_measure_dict(
                ("V_source_sweep_ac", "I_sense"),
                vmax,
                vstep,
                freq,
                high,
                low,
                swpmode,
                "",
                high,
                low,
                wrapper_lst=src_sens_lst,
                compliance_lst=[compliance],
                special_name=folder_name,
                measure_nickname="vi-curve-ac",
            )
        plotobj = DataManipulator(1)

        logger.info("filepath: %s", mea_dict["file_path"])
        logger.info("no of columns(with time column): %d", mea_dict["record_num"])

        if individual_plot:
            plotobj.live_plot_init(
                1,
                1,
                1 if freq is None else 2,
                titles=[[r"$V-I Curve$"]],
                axes_labels=[[[r"$V$", r"$I$"]]],
                line_labels=[[["V-I"]]] if freq is None else [["V-I-x", "V-I-y"]],
            )
            plotobj.start_saving(mea_dict["plot_record_path"], saving_interval)

        for i in mea_dict["gen_lst"]:
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], i)
            if freq is None:
                plotobj.live_plot_update(0, 0, 0, i[1], i[2], incremental=True)
            else:
                plotobj.live_plot_update(
                    [0, 0],
                    [0, 0],
                    [0, 1],
                    [i[1], i[1]],
                    [i[2], i[3]],
                    incremental=True,
                )
            time.sleep(step_time)

        if individual_plot:
            plotobj.stop_saving()

    @task(name="rt", cache_policy=NO_CACHE)
    def measure_VV_II_BTvary_rt_task(
        self,
        *,
        vds: float,
        ds_high: int | str,
        ds_low: int | str,
        ds_meter: Meter | list[Meter],
        ds_compliance: float | str,
        freq: Optional[float] = None,
        vg: float,
        vg_high: int | str,
        vg_meter: Meter,
        vg_compliance: float | str,
        field: float,
        temperature_start: float,
        temperature_end: float,
        folder_name: str = "",
        step_time: float = 0.7,
        wait_before_vary: float = 13,
        vary_loop: bool = False,
        individual_plot: bool = True,
        saving_interval: float = 7,
    ):
        self.measure_VV_II_BTvary_rt(
            vds=vds,
            ds_high=ds_high,
            ds_low=ds_low,
            ds_meter=ds_meter,
            ds_compliance=ds_compliance,
            freq=freq,
            vg=vg,
            vg_high=vg_high,
            vg_meter=vg_meter,
            vg_compliance=vg_compliance,
            field=field,
            temperature_start=temperature_start,
            temperature_end=temperature_end,
            folder_name=folder_name,
            step_time=step_time,
            wait_before_vary=wait_before_vary,
            vary_loop=vary_loop,
            individual_plot=individual_plot,
            saving_interval=saving_interval,
        )

    def measure_VV_II_BTvary_rt(
        self,
        *,
        vds: float,
        ds_high: int | str,
        ds_low: int | str,
        ds_meter: Meter | list[Meter],
        ds_compliance: float | str,
        freq: Optional[float] = None,
        vg: float,
        vg_high: int | str,
        vg_meter: Meter,
        vg_compliance: float | str,
        field: float,
        temperature_start: float,
        temperature_end: float,
        folder_name: str = "",
        step_time: float = 0.7,
        wait_before_vary: float = 13,
        vary_loop: bool = False,
        individual_plot: bool = True,
        saving_interval: float = 7,
    ) -> None:
        """
        measure the V-V and I-I curve using one or two source meters, with other info (B, T, etc.)

        Args:
            vds: float, the drain-source voltage
            ds_high: int | str, the high terminal of the drain-source
            ds_low: int | str, the low terminal of the drain-source
            ds_meter: Meter | list[Meter], the meter used for both source and sense or two meters separately in a list
            ds_compliance: float | str, the compliance of the drain-source meter
            freq: float, the frequency
            vg: float, the gate voltage
            vg_high: int | str, the high terminal of the gate
            vg_meter: Meter, the meter used for the gate
            vg_compliance: float | str, the compliance of the gate meter
            field: float, the field
            temperature_start: float, the start temperature
            temperature_end: float, the end temperature
            folder_name: str, the folder name
            step_time: float, the step time
            wait_before_vary: float, the wait before vary
            vary_loop: bool, the vary loop
            individual_plot: bool, the individual plot
            saving_interval: float, the saving interval in seconds
        """
        if isinstance(ds_meter, list):
            logger.validate(len(ds_meter) == 2, "ds_meter must be a list of two meters")
            ds_src_sens_lst = ds_meter
        else:
            ds_src_sens_lst = [ds_meter, ds_meter]

        plotobj = DataManipulator(1)
        if freq is None:
            mea_dict = self.get_measure_dict(
                (
                    "V_source_fixed_dc",
                    "V_source_fixed_dc",
                    "I_sense",
                    "I_sense",
                    "B_fixed",
                    "T_vary",
                ),
                vds,
                ds_high,
                ds_low,
                vg,
                vg_high,
                0,
                "",
                ds_high,
                ds_low,
                "",
                vg_high,
                0,
                field,
                temperature_start,
                temperature_end,
                wrapper_lst=[
                    ds_src_sens_lst[0],
                    vg_meter,
                    ds_src_sens_lst[1],
                    vg_meter,
                ],
                compliance_lst=[ds_compliance, vg_compliance],
                if_combine_gen=True,  # False for coexistence of vary and mapping
                special_name=folder_name,
                measure_nickname="rt-dc",
                vary_loop=vary_loop,
                wait_before_vary=wait_before_vary,
            )
        else:
            mea_dict = self.get_measure_dict(
                (
                    "V_source_fixed_ac",
                    "V_source_fixed_dc",
                    "I_sense",
                    "I_sense",
                    "B_fixed",
                    "T_vary",
                ),
                vds,
                freq,
                ds_high,
                ds_low,
                vg,
                vg_high,
                0,
                "",
                ds_high,
                ds_low,
                "",
                vg_high,
                0,
                field,
                temperature_start,
                temperature_end,
                wrapper_lst=[
                    ds_src_sens_lst[0],
                    vg_meter,
                    ds_src_sens_lst[1],
                    vg_meter,
                ],
                compliance_lst=[ds_compliance, vg_compliance],
                if_combine_gen=True,  # False for coexistence of vary and mapping
                special_name=folder_name,
                measure_nickname="rt-ac",
                vary_loop=vary_loop,
                wait_before_vary=wait_before_vary,
            )

        logger.info("filepath: %s", mea_dict["file_path"])
        logger.info("no of columns(with time column): %d", mea_dict["record_num"])
        logger.info("vary modules: %s", mea_dict["vary_mod"])

        vary_lst, _, _ = self._extract_vary(mea_dict)
        # modify the plot configuration
        begin_vary = False
        if individual_plot:
            plotobj.live_plot_init(
                1, 3, 1 if freq is None else 2, titles=[["T I_{ds}", "T I_{g}", "t T"]]
            )
            plotobj.start_saving(mea_dict["plot_record_path"], saving_interval)

        for gen_i in mea_dict["gen_lst"]:
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], gen_i)
            if freq is None:
                plotobj.live_plot_update(
                    [0, 0],
                    [0, 1],
                    [0, 0],
                    [gen_i[6], gen_i[6]],
                    [gen_i[3], gen_i[4]],
                    incremental=True,
                )
            else:
                plotobj.live_plot_update(
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [gen_i[9], gen_i[9], gen_i[9]],
                    [gen_i[3], gen_i[4], gen_i[7]],
                    incremental=True,
                )
            plotobj.live_plot_update(0, 2, 0, gen_i[0], gen_i[9], incremental=True)
            time.sleep(step_time)

            if not begin_vary:
                for funci in vary_lst:
                    funci()
                begin_vary = True

        if individual_plot:
            plotobj.stop_saving()

    @task(name="swp-gate", cache_policy=NO_CACHE)
    def measure_VVswp_II_BT_gateswp_task(
        self,
        *,
        vds: float,
        ds_high: int | str,
        ds_low: int | str,
        ds_meter: Meter | list[Meter],
        ds_compliance: float | str,
        freq: Optional[float] = None,
        vg_max: float,
        vg_step: float,
        vg_high: int | str,
        vg_swpmode: str,
        vg_swp_lst: Sequence[float] = None,
        vg_meter: Meter,
        vg_compliance: float | str,
        field: float = 0,
        temperature: float,
        folder_name: str = "",
        step_time: float = 0.5,
        individual_plot: bool = True,
        saving_interval: float = 7,
    ):
        self.measure_VVswp_II_BT_gateswp(
            vds=vds,
            ds_high=ds_high,
            ds_low=ds_low,
            ds_meter=ds_meter,
            ds_compliance=ds_compliance,
            freq=freq,
            vg_max=vg_max,
            vg_step=vg_step,
            vg_high=vg_high,
            vg_swpmode=vg_swpmode,
            vg_swp_lst=vg_swp_lst,
            vg_meter=vg_meter,
            vg_compliance=vg_compliance,
            field=field,
            temperature=temperature,
            folder_name=folder_name,
            step_time=step_time,
            individual_plot=individual_plot,
            saving_interval=saving_interval,
        )

    def measure_VVswp_II_BT_gateswp(
        self,
        *,
        vds: float,
        ds_high: int | str,
        ds_low: int | str,
        ds_meter: Meter | list[Meter],
        ds_compliance: float | str,
        freq: Optional[float] = None,
        vg_max: float,
        vg_step: float,
        vg_high: int | str,
        vg_swpmode: str,
        vg_swp_lst: Sequence[float] = None,
        vg_meter: Meter,
        vg_compliance: float | str,
        field: float = 0,
        temperature: float,
        folder_name: str = None,
        step_time: float = 0.5,
        individual_plot: bool = True,
        saving_interval: float = 7,
    ) -> None:
        """
        measure the Vg-I curve using TWO DC source meters, with other info (B, T, etc.) NOTE the vg_swp_lst will override the vg_step, vg_max and vg_swpmode

        Args:
            vds: float, the drain-source voltage
            ds_high: int | str, the high terminal of the drain-source
            ds_low: int | str, the low terminal of the drain-source
            ds_meter: Meter | list[Meter], the meter used for both source and sense or two meters separately in a list
            ds_compliance: float | str, the compliance of the drain-source meter
            freq: float, the frequency
            vg_max: float, the maximum gate voltage
            vg_step: float, the step gate voltage
            vg_high: int | str, the high terminal of the gate
            vg_swpmode: str, the sweep mode of the gate
            vg_swp_lst: Sequence[float], the list of gate voltages
            vg_meter: Meter, the meter used for the gate
            vg_compliance: float | str, the compliance of the gate meter
            field: float, the field
            temperature: float, the temperature
            folder_name: str, the folder name
            step_time: float, the step time
            individual_plot: bool, the individual plot
            saving_interval: float, the saving interval in seconds
        """
        if isinstance(ds_meter, list):
            logger.validate(len(ds_meter) == 2, "ds_meter must be a list of two meters")
            ds_src_sens_lst = ds_meter
        else:
            ds_src_sens_lst = [ds_meter, ds_meter]

        plotobj = DataManipulator(1)
        if vg_swp_lst is not None:
            vg_swpmode = "manual"
            swp_lst = [vg_swp_lst]
        else:
            swp_lst = None

        if freq is None:
            mea_dict = self.get_measure_dict(
                (
                    "V_source_fixed_dc",
                    "V_source_sweep_dc",
                    "I_sense",
                    "I_sense",
                    "B_fixed",
                    "T_fixed",
                ),
                vds,
                ds_high,
                ds_low,
                vg_max,
                vg_step,
                vg_high,
                0,
                vg_swpmode,
                "",
                ds_high,
                ds_low,
                "",
                vg_high,
                0,
                field,
                temperature,
                wrapper_lst=[
                    ds_src_sens_lst[0],
                    vg_meter,
                    ds_src_sens_lst[1],
                    vg_meter,
                ],
                compliance_lst=[ds_compliance, vg_compliance],
                if_combine_gen=True,  # False for coexistence of vary and mapping
                special_name=folder_name,
                sweep_tables=swp_lst,
                measure_nickname="swp-gate-dc",
            )
        else:
            mea_dict = self.get_measure_dict(
                (
                    "V_source_fixed_ac",
                    "V_source_sweep_dc",
                    "I_sense",
                    "I_sense",
                    "B_fixed",
                    "T_fixed",
                ),
                vds,
                freq,
                ds_high,
                ds_low,
                vg_max,
                vg_step,
                vg_high,
                0,
                vg_swpmode,
                "",
                ds_high,
                ds_low,
                "",
                vg_high,
                0,
                field,
                temperature,
                wrapper_lst=[
                    ds_src_sens_lst[0],
                    vg_meter,
                    ds_src_sens_lst[1],
                    vg_meter,
                ],
                compliance_lst=[ds_compliance, vg_compliance],
                if_combine_gen=True,  # False for coexistence of vary and mapping
                special_name=folder_name,
                sweep_tables=swp_lst,
                measure_nickname="swp-gate-ac",
            )

        logger.info("filepath: %s", mea_dict["file_path"])
        logger.info("no of columns(with time column): %d", mea_dict["record_num"])
        logger.info("vary modules: %s", mea_dict["vary_mod"])

        # modify the plot configuration
        # note i[0] is timer
        if individual_plot:
            plotobj.live_plot_init(
                1, 2, 1 if freq is None else 2, titles=[[r"$R V_g$", r"$I_g V_g$"]]
            )
            plotobj.start_saving(mea_dict["plot_record_path"], saving_interval)

        for i in mea_dict["gen_lst"]:
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], i)
            if freq is None:
                plotobj.live_plot_update(
                    [0, 0],
                    [0, 1],
                    [0, 0],
                    [i[2], i[2]],
                    [vds / i[3], i[4]],
                    incremental=True,
                )
            else:
                plotobj.live_plot_update(
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [i[2], i[2], i[2]],
                    [vds / i[3], vds / i[4], i[7]],
                    incremental=True,
                )
            time.sleep(step_time)

        if individual_plot:
            plotobj.stop_saving()

    @task(name="vi-curve", cache_policy=NO_CACHE)
    def measure_VswpV_II_BT_vicurve_task(
        self,
        *,
        vds_max: float,
        vds_step: float,
        freq: Optional[float] = None,
        ds_high: int | str,
        ds_low: int | str,
        vds_swpmode: str,
        vds_swp_lst: Sequence[float] = None,
        ds_meter: Meter | list[Meter],
        ds_compliance: float | str,
        vg: float,
        vg_high: int | str,
        vg_meter: Meter,
        vg_compliance: float | str,
        field: float = 0,
        temperature: float,
        folder_name: str = None,
        step_time: float = 0.3,
        individual_plot: bool = True,
        saving_interval: float = 7,
    ):
        self.measure_VswpV_II_BT_vicurve(
            vds_max=vds_max,
            vds_step=vds_step,
            freq=freq,
            ds_high=ds_high,
            ds_low=ds_low,
            vds_swpmode=vds_swpmode,
            vds_swp_lst=vds_swp_lst,
            ds_meter=ds_meter,
            ds_compliance=ds_compliance,
            vg=vg,
            vg_high=vg_high,
            vg_meter=vg_meter,
            vg_compliance=vg_compliance,
            field=field,
            temperature=temperature,
            folder_name=folder_name,
            step_time=step_time,
            individual_plot=individual_plot,
            saving_interval=saving_interval,
        )

    def measure_VswpV_II_BT_vicurve(
        self,
        *,
        vds_max: float,
        vds_step: float,
        freq: Optional[float] = None,
        ds_high: int | str,
        ds_low: int | str,
        vds_swpmode: str,
        vds_swp_lst: Sequence[float] = None,
        ds_meter: Meter | list[Meter],
        ds_compliance: float | str,
        vg: float,
        vg_high: int | str,
        vg_meter: Meter,
        vg_compliance: float | str,
        field: float = 0,
        temperature: float,
        folder_name: str = None,
        step_time: float = 0.3,
        individual_plot: bool = True,
        saving_interval: float = 7,
    ):
        """
        measure the Vds-I curve using TWO DC source meters, with other info (B, T, etc.) NOTE the vds_swp_lst will override the vds_step, vds_max and vds_swpmode

        Args:
            vds_max: float, the maximum drain-source voltage
            vds_step: float, the step drain-source voltage
            freq: float, the frequency
            ds_high: int | str, the high terminal of the drain-source
            ds_low: int | str, the low terminal of the drain-source
            vds_swpmode: str, the sweep mode of the drain-source
            vds_swp_lst: Sequence[float], the list of drain-source voltages
            ds_meter: Meter | list[Meter], the meter used for both source and sense or two meters separately in a list
            ds_compliance: float | str, the compliance of the drain-source meter
            vg: float, the gate voltage
            vg_high: int | str, the high terminal of the gate
            vg_meter: Meter, the meter used for the gate
            vg_compliance: float | str, the compliance of the gate meter
            field: float, the field
            temperature: float, the temperature
            folder_name: str, the folder name
            step_time: float, the step time
            individual_plot: bool, the individual plot
            saving_interval: float, the saving interval in seconds
        """
        if isinstance(ds_meter, list):
            logger.validate(len(ds_meter) == 2, "ds_meter must be a list of two meters")
            ds_src_sens_lst = ds_meter
        else:
            ds_src_sens_lst = [ds_meter, ds_meter]

        plotobj = DataManipulator(1)
        if vds_swp_lst is not None:
            vds_swpmode = "manual"

        if freq is None:
            mea_dict = self.get_measure_dict(
                (
                    "V_source_sweep_dc",
                    "V_source_fixed_dc",
                    "I_sense",
                    "I_sense",
                    "B_fixed",
                    "T_fixed",
                ),
                vds_max,
                vds_step,
                ds_high,
                ds_low,
                vds_swpmode,
                vg,
                vg_high,
                0,
                "",
                ds_high,
                ds_low,
                "",
                vg_high,
                0,
                field,
                temperature,
                wrapper_lst=[
                    ds_src_sens_lst[0],
                    vg_meter,
                    ds_src_sens_lst[1],
                    vg_meter,
                ],
                compliance_lst=[ds_compliance, vg_compliance],
                if_combine_gen=True,  # False for coexistence of vary and mapping
                special_name=folder_name,
                sweep_tables=[vds_swp_lst],
                measure_nickname="swpds-dc",
            )
        else:
            mea_dict = self.get_measure_dict(
                (
                    "V_source_sweep_ac",
                    "V_source_fixed_dc",
                    "I_sense",
                    "I_sense",
                    "B_fixed",
                    "T_fixed",
                ),
                vds_max,
                vds_step,
                freq,
                ds_high,
                ds_low,
                vds_swpmode,
                vg,
                vg_high,
                0,
                "",
                ds_high,
                ds_low,
                "",
                vg_high,
                0,
                field,
                temperature,
                wrapper_lst=[
                    ds_src_sens_lst[0],
                    vg_meter,
                    ds_src_sens_lst[1],
                    vg_meter,
                ],
                compliance_lst=[ds_compliance, vg_compliance],
                if_combine_gen=True,  # False for coexistence of vary and mapping
                special_name=folder_name,
                sweep_tables=[vds_swp_lst],
                measure_nickname="swpds-ac",
            )

        logger.info("filepath: %s", mea_dict["file_path"])
        logger.info("no of columns(with time column): %d", mea_dict["record_num"])
        logger.info("vary modules: %s", mea_dict["vary_mod"])

        # modify the plot configuration
        # note i[0] is timer
        if individual_plot:
            plotobj.live_plot_init(
                1, 2, 1 if freq is None else 2, titles=[[r"V_{ds} I", r"V_{ds} T"]]
            )
            plotobj.start_saving(mea_dict["plot_record_path"], saving_interval)

        for i in mea_dict["gen_lst"]:
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], i)
            if freq is None:
                plotobj.live_plot_update(
                    [0, 0],
                    [0, 1],
                    [0, 0],
                    [i[1], i[1]],
                    [i[3], i[6]],
                    incremental=True,
                )
            else:
                plotobj.live_plot_update(
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [i[1], i[1], i[1]],
                    [i[3], i[4], i[9]],
                    incremental=True,
                )
            time.sleep(step_time)

        if individual_plot:
            plotobj.stop_saving()

    @task(name="rh-loop", cache_policy=NO_CACHE)
    def measure_VV_II_BvaryT_rhloop_task(
        self,
        *,
        vds: float,
        ds_high: int | str,
        ds_low: int | str,
        ds_meter: Meter | list[Meter],
        ds_compliance: float | str,
        freq: Optional[float] = None,
        vg: float,
        vg_high: int | str,
        vg_meter: Meter,
        vg_compliance: float | str,
        field_start: float,
        field_end: float,
        temperature: float,
        folder_name: str = None,
        step_time: float = 0.3,
        wait_before_vary: float = 5,
        vary_loop: bool = True,
        individual_plot: bool = True,
        saving_interval: float = 7,
    ):
        self.measure_VV_II_BvaryT_rhloop(
            vds=vds,
            ds_high=ds_high,
            ds_low=ds_low,
            ds_meter=ds_meter,
            ds_compliance=ds_compliance,
            freq=freq,
            vg=vg,
            vg_high=vg_high,
            vg_meter=vg_meter,
            vg_compliance=vg_compliance,
            field_start=field_start,
            field_end=field_end,
            temperature=temperature,
            folder_name=folder_name,
            step_time=step_time,
            wait_before_vary=wait_before_vary,
            vary_loop=vary_loop,
            individual_plot=individual_plot,
            saving_interval=saving_interval,
        )

    def measure_VV_II_BvaryT_rhloop(
        self,
        *,
        vds: float,
        ds_high: int | str,
        ds_low: int | str,
        ds_meter: Meter | list[Meter],
        ds_compliance: float | str,
        freq: Optional[float] = None,
        vg: float,
        vg_high: int | str,
        vg_meter: Meter,
        vg_compliance: float | str,
        field_start: float,
        field_end: float,
        temperature: float,
        folder_name: str = None,
        step_time: float = 0.3,
        wait_before_vary: float = 5,
        vary_loop: bool = True,
        individual_plot: bool = True,
        saving_interval: float = 7,
    ):
        """
        measure the V-V and I-I curve using one or two source meters, with other info (B, T, etc.)

        Args:
            vds: float, the drain-source voltage
            ds_high: int | str, the high terminal of the drain-source
            ds_low: int | str, the low terminal of the drain-source
            ds_meter: Meter | list[Meter], the meter used for both source and sense or two meters separately in a list
            ds_compliance: float | str, the compliance of the drain-source meter
            freq: float, the frequency
            vg: float, the gate voltage
            vg_high: int | str, the high terminal of the gate
            vg_meter: Meter, the meter used for the gate
            vg_compliance: float | str, the compliance of the gate meter
            field_start: float, the start field
            field_end: float, the end field
            temperature: float, the temperature
            folder_name: str, the folder name
            step_time: float, the step time
            wait_before_vary: float, the wait before vary
            vary_loop: bool, the vary loop
            individual_plot: bool, the individual plot
            saving_interval: float, the saving interval in seconds
        """
        if isinstance(ds_meter, list):
            logger.validate(len(ds_meter) == 2, "ds_meter must be a list of two meters")
            ds_src_sens_lst = ds_meter
        else:
            ds_src_sens_lst = [ds_meter, ds_meter]

        plotobj = DataManipulator(1)
        if freq is None:
            mea_dict = self.get_measure_dict(
                (
                    "V_source_fixed_dc",
                    "V_source_fixed_dc",
                    "I_sense",
                    "I_sense",
                    "B_vary",
                    "T_fixed",
                ),
                vds,
                ds_high,
                ds_low,
                vg,
                vg_high,
                0,
                "",
                ds_high,
                ds_low,
                "",
                vg_high,
                0,
                field_start,
                field_end,
                temperature,
                wrapper_lst=[
                    ds_src_sens_lst[0],
                    vg_meter,
                    ds_src_sens_lst[1],
                    vg_meter,
                ],
                compliance_lst=[ds_compliance, vg_compliance],
                if_combine_gen=True,  # False for coexistence of vary and mapping
                special_name=folder_name,
                measure_nickname="rh-loop-dc",
                vary_loop=vary_loop,
                wait_before_vary=wait_before_vary,
            )
        else:
            mea_dict = self.get_measure_dict(
                (
                    "V_source_fixed_ac",
                    "V_source_fixed_dc",
                    "I_sense",
                    "I_sense",
                    "B_vary",
                    "T_fixed",
                ),
                vds,
                freq,
                ds_high,
                ds_low,
                vg,
                vg_high,
                0,
                "",
                ds_high,
                ds_low,
                "",
                vg_high,
                0,
                field_start,
                field_end,
                temperature,
                wrapper_lst=[
                    ds_src_sens_lst[0],
                    vg_meter,
                    ds_src_sens_lst[1],
                    vg_meter,
                ],
                compliance_lst=[ds_compliance, vg_compliance],
                if_combine_gen=True,  # False for coexistence of vary and mapping
                special_name=folder_name,
                measure_nickname="rh-loop-ac",
                vary_loop=vary_loop,
            )

        logger.info("filepath: %s", mea_dict["file_path"])
        logger.info("no of columns(with time column): %d", mea_dict["record_num"])
        logger.info("vary modules: %s", mea_dict["vary_mod"])

        vary_lst, _, _ = self._extract_vary(mea_dict)
        # modify the plot configuration
        begin_vary = False
        counter = 0
        if individual_plot:
            plotobj.live_plot_init(
                1, 3, 1 if freq is None else 2, titles=[["B I", "B T", "t B"]]
            )
            plotobj.start_saving(mea_dict["plot_record_path"], saving_interval)

        for gen_i in mea_dict["gen_lst"]:
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], gen_i)
            if freq is None:
                plotobj.live_plot_update(
                    [0, 0],
                    [0, 1],
                    [0, 0],
                    [gen_i[5], gen_i[5]],
                    [gen_i[3], gen_i[6]],
                    incremental=True,
                )
                plotobj.live_plot_update(0, 2, 0, gen_i[0], gen_i[5], incremental=True)
            else:
                plotobj.live_plot_update(
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [gen_i[8], gen_i[8], gen_i[8]],
                    [gen_i[3], gen_i[4], gen_i[9]],
                    incremental=True,
                )
                plotobj.live_plot_update(0, 2, 0, gen_i[0], gen_i[8], incremental=True)
            time.sleep(step_time)

            if not begin_vary:
                for funci in vary_lst:
                    funci()
                begin_vary = True

        if individual_plot:
            plotobj.stop_saving()

    def measure_VV_IwI2wI_BvaryT_rhloop(
        self,
        *,
        vds: float,
        ds_high: int | str,
        ds_low: int | str,
        ds_meter_source: Meter,
        ds_meter_1w: Meter,
        ds_meter_2w: Meter,
        ds_compliance: float | str,
        freq: float,
        vg: float,
        vg_high: int | str,
        vg_meter: Meter,
        vg_compliance: float | str,
        field_start: float,
        field_end: float,
        temperature: float,
        folder_name: str = None,
        step_time: float = 0.3,
        wait_before_vary: float = 5,
        vary_loop: bool = True,
        individual_plot: bool = True,
        saving_interval: float = 7,
    ):
        """
        measure the V-V and I-I curve using one or two source meters, with other info (B, T, etc.)

        Args:
            vds: float, the drain-source voltage
            ds_high: int | str, the high terminal of the drain-source
            ds_low: int | str, the low terminal of the drain-source
            ds_meter_source: Meter, the meter used for source
            ds_meter_1w: Meter, the meter used for the 1w signal
            ds_meter_2w: Meter, the meter used for the 2w signal
            ds_compliance: float | str, the compliance of the drain-source meter
            freq: float, the frequency
            vg: float, the gate voltage
            vg_high: int | str, the high terminal of the gate
            vg_meter: Meter, the meter used for the gate
            vg_compliance: float | str, the compliance of the gate meter
            field_start: float, the start field
            field_end: float, the end field
            temperature: float, the temperature
            folder_name: str, the folder name
            step_time: float, the step time
            wait_before_vary: float, the wait before vary
            vary_loop: bool, the vary loop
            individual_plot: bool, the individual plot
            saving_interval: float, the saving interval in seconds
        """
        plotobj = DataManipulator(1)
        mea_dict = self.get_measure_dict(
            (
                "V_source_fixed_ac",
                "V_source_fixed_dc",
                "I_sense",
                "I_sense",
                "I_sense",
                "B_vary",
                "T_fixed",
            ),
            vds,
            freq,
            ds_high,
            ds_low,
            vg,
            vg_high,
            0,
            "1w",
            ds_high,
            ds_low,
            "2w",
            ds_high,
            ds_low,
            "gate",
            vg_high,
            0,
            field_start,
            field_end,
            temperature,
            wrapper_lst=[
                ds_meter_source,
                vg_meter,
                ds_meter_1w,
                ds_meter_2w,
                vg_meter,
            ],
            compliance_lst=[ds_compliance, vg_compliance],
            if_combine_gen=True,  # False for coexistence of vary and mapping
            special_name=folder_name,
            measure_nickname="rh-loop-ac",
            vary_loop=vary_loop,
            wait_before_vary=wait_before_vary,
        )

        logger.info("filepath: %s", mea_dict["file_path"])
        logger.info("no of columns(with time column): %d", mea_dict["record_num"])
        logger.info("vary modules: %s", mea_dict["vary_mod"])

        ds_meter_1w.reference_set(harmonic=1)
        ds_meter_2w.reference_set(harmonic=2)

        vary_lst, _, _ = self._extract_vary(mea_dict)
        # modify the plot configuration
        begin_vary = False
        if individual_plot:
            plotobj.live_plot_init(
                2, 2, 2, titles=[["B I1w", "B I2w"], ["B T", "t B"]]
            )
            plotobj.start_saving(mea_dict["plot_record_path"], saving_interval)

        for gen_i in mea_dict["gen_lst"]:
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], gen_i)
            plotobj.live_plot_update(
                [0, 0, 0, 0, 1],
                [0, 0, 1, 1, 0],
                [0, 1, 0, 1, 0],
                [gen_i[12], gen_i[12], gen_i[12], gen_i[12], gen_i[12]],
                [gen_i[3],  gen_i[4],  gen_i[7], gen_i[8], gen_i[13]],
                incremental=True,
            )
            plotobj.live_plot_update(1, 1, 0, gen_i[0], gen_i[12], incremental=True)
            time.sleep(step_time)

            if not begin_vary:
                for funci in vary_lst:
                    funci()
                begin_vary = True

        if individual_plot:
            plotobj.stop_saving()


    @task(name="ds-gate-mapping", cache_policy=NO_CACHE)
    def measure_VswpVswp_II_BT_dsgatemapping_task(
        self,
        *,
        constrained: bool = False,
        vds_max: float,
        ds_map_lst: Sequence[float],
        ds_high: int | str,
        ds_low: int | str,
        ds_meter: Meter | list[Meter],
        ds_compliance: float | str,
        freq: Optional[float] = None,
        vg: float,
        gate_map_lst: Sequence[float],
        vg_high: int | str,
        vg_meter: Meter,
        vg_compliance: float | str,
        field: float,
        temperature: float,
        folder_name: str = None,
        step_time: float = 1,
        individual_plot: bool = True,
        ds_gate_order: tuple[int, int] = (0, 1),
        saving_interval: float = 7,
    ):
        self.measure_VswpVswp_II_BT_dsgatemapping(
            constrained=constrained,
            vds_max=vds_max,
            ds_map_lst=ds_map_lst,
            ds_high=ds_high,
            ds_low=ds_low,
            ds_meter=ds_meter,
            ds_compliance=ds_compliance,
            freq=freq,
            vg=vg,
            gate_map_lst=gate_map_lst,
            vg_high=vg_high,
            vg_meter=vg_meter,
            vg_compliance=vg_compliance,
            field=field,
            temperature=temperature,
            folder_name=folder_name,
            step_time=step_time,
            individual_plot=individual_plot,
            ds_gate_order=ds_gate_order,
            saving_interval=saving_interval,
        )

    def measure_VswpVswp_II_BT_dsgatemapping(
        self,
        *,
        constrained: bool = False,
        vds_max: float,
        freq: Optional[float] = None,
        ds_map_lst: Sequence[float],
        ds_high: int | str,
        ds_low: int | str,
        ds_meter: Meter | list[Meter],
        ds_compliance: float | str,
        vg: float,
        gate_map_lst: Sequence[float],
        vg_high: int | str,
        vg_meter: Meter,
        vg_compliance: float | str,
        field: float,
        temperature: float,
        folder_name: str = None,
        step_time: float = 1,
        individual_plot: bool = True,
        ds_gate_order: tuple[int, int] = (0, 1),
        calculate_from_ds: Optional[Callable] = None,
        contour_ac: Literal["X", "Y", "R", "Theta"] = "R",
        saving_interval: float = 7,
    ):
        """
        NOTE: the sweep order is important, for order (0,1), it means the inner loop is vg, the outer loop is vds (if not constrained)

        Args:
            freq: Optional[float], the frequency
            ds_meter: Meter | list[Meter], the drain-source meter
            ds_compliance: float | str, the compliance of the drain-source meter
            vg_meter: Meter, the gate meter
            vg_compliance: float | str, the compliance of the gate meter
            field: float, the field
            temperature: float, the temperature
            folder_name: str, the folder name
            step_time: float, the step time
            individual_plot: bool, the individual plot
            ds_gate_order: tuple[int, int], the drain-source gate order
            calculate_from_ds: Callable, the function to calculate the gate voltage from the drain-source voltage
            constrained: bool, the constrained
            vds_max: float, the drain-source voltage max
            ds_map_lst: Sequence[float], the drain-source voltage map list
            ds_high: int | str, the drain-source high
            ds_low: int | str, the drain-source low
            vg: float, the gate voltage
            gate_map_lst: Sequence[float], the gate voltage map list
            vg_high: int | str, the gate high
            vg_meter: Meter, the gate meter
            vg_compliance: float | str, the compliance of the gate meter
            field: float, the field
            temperature: float, the temperature
            folder_name: str, the folder name
            step_time: float, the step time
            individual_plot: bool, the individual plot
            ds_gate_order: tuple[int, int], the drain-source gate order
            calculate_from_ds: Callable, the function to calculate the targeted property from the drain-source voltage
            contour_ac: Literal["X", "Y", "R", "Theta"], the ac sense property used for contour plot
            saving_interval: float, the saving interval in seconds
        """
        contour_ac_idx = {"X": 3, "Y": 4, "R": 5, "Theta": 6}[contour_ac]
        if isinstance(ds_meter, list):
            logger.validate(len(ds_meter) == 2, "ds_meter must be a list of two meters")
            ds_src_sens_lst = ds_meter
        else:
            ds_src_sens_lst = [ds_meter, ds_meter]

        plotobj = DataManipulator(1)
        if not constrained:
            map_lsts = self.create_mapping(ds_map_lst, gate_map_lst, idxs=ds_gate_order)
            # =======only applied to ds and vg mapping=========
            if_inner_loop_is_ds = ds_gate_order[0] == 1
            inner_loop_len = (
                len(ds_map_lst) if if_inner_loop_is_ds else len(gate_map_lst)
            )
            # ==================================================
            if calculate_from_ds is not None:
                logger.warning("calculate_fromds causes no effect when not constrained")
        else:
            if calculate_from_ds is None:
                calculate_from_ds = lambda x: x
            map_lsts = [ds_map_lst, gate_map_lst]

        # Core configuration
        # Generate the measurement generator
        if freq is None:
            mea_dict = self.get_measure_dict(
                (
                    "V_source_sweep_dc",
                    "V_source_sweep_dc",
                    "I_sense",
                    "I_sense",
                    "B_fixed",
                    "T_fixed",
                ),
                vds_max,
                0,
                ds_high,
                ds_low,
                "manual",
                vg,
                0,
                vg_high,
                0,
                "manual",
                "",
                ds_high,
                ds_low,
                "",
                vg_high,
                0,
                field,
                temperature,
                wrapper_lst=[
                    ds_src_sens_lst[0],
                    vg_meter,
                    ds_src_sens_lst[1],
                    vg_meter,
                ],
                compliance_lst=[ds_compliance, vg_compliance],
                sr830_current_resistor=None,  # only float
                if_combine_gen=True,  # False for coexistence of vary and mapping
                special_name=folder_name,
                sweep_tables=map_lsts,
                measure_nickname="ds-gate-mapping-dc",
            )
        else:
            mea_dict = self.get_measure_dict(
                (
                    "V_source_sweep_ac",
                    "V_source_sweep_dc",
                    "I_sense",
                    "I_sense",
                    "B_fixed",
                    "T_fixed",
                ),
                vds_max,
                0,
                freq,
                ds_high,
                ds_low,
                "manual",
                vg,
                0,
                vg_high,
                0,
                "manual",
                "",
                ds_high,
                ds_low,
                "",
                vg_high,
                0,
                field,
                temperature,
                wrapper_lst=[
                    ds_src_sens_lst[0],
                    vg_meter,
                    ds_src_sens_lst[1],
                    vg_meter,
                ],
                compliance_lst=[ds_compliance, vg_compliance],
                sr830_current_resistor=None,  # only float
                if_combine_gen=True,  # False for coexistence of vary and mapping
                special_name=folder_name,
                sweep_tables=map_lsts,
                measure_nickname="ds-gate-mapping-ac",
            )

        logger.info("filepath: %s", mea_dict["file_path"])
        logger.info("no of columns(with time column): %d", mea_dict["record_num"])
        logger.info("vary modules: %s", mea_dict["vary_mod"])
        # modify the plot configuration
        if individual_plot:
            if not constrained:
                titles = (
                    [["V_{ds} I_{ds}"], ["V_{ds} I_{g}"], ["contour"]]
                    if if_inner_loop_is_ds
                    else [["V_{g} I_{ds}"], ["V_{g} I_{g}"], ["contour"]]
                )
                axes_labels = (
                    [
                        [[r"$V_{ds}$", r"$I_{ds}$"]],
                        [[r"$V_{ds}$", r"$I_{g}$"]],
                        [r"$V_{ds}$", r"$V_{g}$"],
                    ]
                    if if_inner_loop_is_ds
                    else [
                        [[r"$V_{g}$", r"$I_{ds}$"]],
                        [[r"$V_{g}$", r"$I_{g}$"]],
                        [r"$V_{ds}$", r"$V_{g}$"],
                    ]
                )
                plotobj.live_plot_init(
                    3,
                    1,
                    1 if freq is None else 2,
                    plot_types=[["scatter"], ["scatter"], ["contour"]],
                    titles=titles,
                    axes_labels=axes_labels,
                )
            else:
                plotobj.live_plot_init(
                    2,
                    1,
                    1 if freq is None else 2,
                    plot_types=[["scatter"], ["scatter"]],
                    titles=[["n I_{ds}"], ["V_{g} I_{g}"]],
                    axes_labels=[
                        [[r"$n$", r"$I_{ds}$"]],
                        [[r"$V_{g}$", r"$I_{ds}$"]],
                    ],
                )

            plotobj.start_saving(mea_dict["plot_record_path"], saving_interval)

        for i in mea_dict["gen_lst"]:
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], i)
            if not constrained:
                x_data = i[1] if if_inner_loop_is_ds else i[2]
                if freq is None:
                    plotobj.live_plot_update(
                        [0, 1],
                        [0] * 2,
                        [0] * 2,
                        [x_data, x_data],
                        [i[3], i[4]],
                        incremental=True,
                        max_points=inner_loop_len,
                    )
                    plotobj.live_plot_update(
                        2,
                        0,
                        0,
                        i[1],
                        i[2],
                        i[3],
                        incremental=True,
                    )
                else:
                    plotobj.live_plot_update(
                        [0, 0, 1],
                        [0] * 3,
                        [0, 1, 0],
                        [x_data, x_data, x_data],
                        [i[3], i[4], i[7]],
                        incremental=True,
                        max_points=inner_loop_len,
                    )
                    plotobj.live_plot_update(
                        2,
                        0,
                        0,
                        i[1],
                        i[2],
                        i[contour_ac_idx],
                        incremental=True,
                    )
            else:
                if freq is None:
                    plotobj.live_plot_update(
                        [0, 1],
                        [0] * 2,
                        [0] * 2,
                        [calculate_from_ds(i[1]), i[2]],
                        [i[3], i[4]],
                        incremental=True,
                    )
                else:
                    plotobj.live_plot_update(
                        [0, 0, 1],
                        [0] * 3,
                        [0, 1, 0],
                        [calculate_from_ds(i[1]), calculate_from_ds(i[1]), i[2]],
                        [i[3], i[4], i[7]],
                        incremental=True,
                    )
            time.sleep(step_time)

        if individual_plot:
            plotobj.stop_saving()

    def _extract_vary(
        self, mea_dict: dict
    ) -> tuple[list[Callable], list[Callable], list[Callable]]:
        """
        extract the vary functions, current value getters and set value getters for using in varying cases

        Currently the vary number is limited to 1
        """

        vary_lst = []
        curr_val_lst = []
        set_val_lst = []
        for i in mea_dict["vary_mod"]:
            match i:
                case "T":
                    vary_lst.append(mea_dict["tmp_vary"][0])
                    curr_val_lst.append(lambda: self.instrs["itc"].temperature)
                    set_val_lst.append(lambda: self.instrs["itc"].temperature_set)
                case "B":
                    vary_lst.append(mea_dict["mag_vary"][0])
                    curr_val_lst.append(lambda: self.instrs["ips"].field)
                    set_val_lst.append(lambda: self.instrs["ips"].field_set)
                case "Theta":
                    vary_lst.append(mea_dict["angle_vary"][0])
                    curr_val_lst.append(self.instrs["rotator"].curr_angle)
                    set_val_lst.append(lambda: self.instrs["rotator"].angle_set)

        logger.validate(
            len(curr_val_lst) == 1, "vary_lst and curr_val_lst have lengths as 1"
        )
        logger.validate(len(vary_lst) == 1, "only one varying parameter is allowed")

        return vary_lst, curr_val_lst, set_val_lst
