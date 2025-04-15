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
from pyomnix.utils import next_lst_gen, convert_unit

logger = get_logger(__name__)


class MeasureFlow(MeasureManager):
    """
    This class is a subclass of MeasureManager and is responsible for managing the measure-related folders and data
    """

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
        plotobj: DataManipulator = None,
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
            if_plot: bool, the individual plot
            saving_interval: float, the saving interval in seconds
        """
        if isinstance(meter, list):
            logger.validate(len(meter) == 2, "meter must be a list of two meters")
            src_sens_lst = meter
        else:
            src_sens_lst = [meter, meter]
        if freq is None:
            mea_dict = self.get_measure_dict(
                ("V_source_sweep_dc", "I_sense_dc"),
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
                ("V_source_sweep_ac", "I_sense_ac"),
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
        if plotobj is None and if_plot:
            plotobj = DataManipulator(1)

        logger.info("filepath: %s", mea_dict["file_path"])
        logger.info("no of columns(with time column): %d", mea_dict["record_num"])

        if if_plot:
            plotobj.live_plot_init(
                1,
                1,
                1 if freq is None else 2,
                titles=[[r"$V-I Curve$"]],
                axes_labels=[[[r"$V$", r"$I$"]]],
                line_labels=[[["V-I"]]] if freq is None else [[["V-I-x", "V-I-y"]]],
            )
            plotobj.start_saving(mea_dict["plot_record_path"], saving_interval)

        for i in mea_dict["gen_lst"]:
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], i)
            time.sleep(step_time)
            if plotobj is None:
                continue
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

        if if_plot:
            plotobj.stop_saving()

    def measure_Vswp_V_vrcurve_lockin(
        self,
        *,
        resistor: float | str,
        vmax: float,
        vstep: float,
        freq: float = None,
        high: int | str,
        low: int | str,
        swpmode: str,
        meter: Meter | list[Meter],
        compliance: float,
        folder_name: str = "",
        step_time: float = 0.3,
        if_plot: bool = True,
        saving_interval: float = 7,
        plotobj: DataManipulator = None,
        source_wait: float = 0.5,
    ) -> None:
        """
        measure the V-R curve using ONE DC source meter, no other info (B, T, etc.). Use freq to indicate ac measurement

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
            if_plot: bool, the individual plot
            saving_interval: float, the saving interval in seconds
        """
        resistor = convert_unit(resistor, "Ohm")[0]
        if isinstance(meter, list):
            logger.validate(len(meter) == 2, "meter must be a list of two meters")
            src_sens_lst = meter
        else:
            src_sens_lst = [meter, meter]
        mea_dict = self.get_measure_dict(
            ("V_source_sweep_ac", "I_sense_ac"),
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
            special_name=f"{resistor}Ohm-{folder_name}",
            measure_nickname="vi-curve-lockin",
            source_wait=source_wait,
        )
        if plotobj is None and if_plot:
            plotobj = DataManipulator(1)

        logger.info("filepath: %s", mea_dict["file_path"])
        logger.info("no of columns(with time column): %d", mea_dict["record_num"])

        if if_plot:
            plotobj.live_plot_init(
                2,
                1,
                2,
                titles=[[r"$V-R Curve$"],
                        [r"$V-V lock-in$"]],
                axes_labels=[[[r"$V$", r"$R$"]],
                              [[r"$V$", r"$V_{lockin}$"]]],
                line_labels=[[["", ""]],
                             [["V-V-x", "V-V-y"]]],
            )
            plotobj.start_saving(mea_dict["plot_record_path"], saving_interval)

        for i in mea_dict["gen_lst"]:
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], i)
            time.sleep(step_time)
            if plotobj is None:
                continue
            plotobj.live_plot_update(
                [0, 1, 1],
                [0, 0, 0],
                [0, 0, 1],
                [i[1], i[1], i[1]],
                [i[4] / (i[1] / resistor), i[2], i[3]],
                incremental=True,
            )

        if if_plot:
            plotobj.stop_saving()

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
        if_plot: bool = True,
        saving_interval: float = 7,
        no_start_vary: bool = True,
        plotobj: Optional[DataManipulator] = None,
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
            if_plot: bool, the individual plot
            saving_interval: float, the saving interval in seconds
        """
        if isinstance(ds_meter, list):
            logger.validate(len(ds_meter) == 2, "ds_meter must be a list of two meters")
            ds_src_sens_lst = ds_meter
        else:
            ds_src_sens_lst = [ds_meter, ds_meter]

        if plotobj is None and if_plot:
            plotobj = DataManipulator(1)
        if freq is None:
            mea_dict = self.get_measure_dict(
                (
                    "V_source_fixed_dc",
                    "V_source_fixed_dc",
                    "I_sense_dc",
                    "I_sense_dc",
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
                no_start_vary=no_start_vary,
            )
        else:
            mea_dict = self.get_measure_dict(
                (
                    "V_source_fixed_ac",
                    "V_source_fixed_dc",
                    "I_sense_ac",
                    "I_sense_dc",
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
                no_start_vary=no_start_vary,
            )

        logger.info("filepath: %s", mea_dict["file_path"])
        logger.info("no of columns(with time column): %d", mea_dict["record_num"])
        logger.info("vary modules: %s", mea_dict["vary_mod"])

        vary_lst, _, _, _ = self._extract_vary(mea_dict)
        # modify the plot configuration
        begin_vary = False
        if if_plot:
            plotobj.live_plot_init(
                1, 3, 1 if freq is None else 2, titles=[["T I_{ds}", "T I_{g}", "t T"]]
            )
            plotobj.start_saving(mea_dict["plot_record_path"], saving_interval)

        for gen_i in mea_dict["gen_lst"]:
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], gen_i)
            time.sleep(step_time)
            if plotobj is None:
                continue
            if freq is None:
                plotobj.live_plot_update(
                    [0, 0],
                    [0, 1],
                    [0, 0],
                    [gen_i[6], gen_i[6]],
                    [gen_i[3], gen_i[4]],
                    incremental=True,
                )
                plotobj.live_plot_update(0, 2, 0, gen_i[0], gen_i[6], incremental=True)
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

            if not begin_vary:
                for funci in vary_lst:
                    funci()
                begin_vary = True

        if if_plot:
            plotobj.stop_saving()

    def measure_VV_V1wI_BTvary_rt_lockin(
        self,
        *,
        resistor: float | str,
        vds: float,
        ds_high: int | str,
        ds_low: int | str,
        ds_meter: Meter | list[Meter],
        ds_compliance: float | str,
        freq: float,
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
        if_plot: bool = True,
        saving_interval: float = 7,
        no_start_vary: bool = True,
        plotobj: Optional[DataManipulator] = None,
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
            if_plot: bool, the individual plot
            saving_interval: float, the saving interval in seconds
        """
        resistor = convert_unit(resistor, "Ohm")[0]
        if isinstance(ds_meter, list):
            logger.validate(len(ds_meter) == 2, "ds_meter must be a list of two meters")
            ds_src_sens_lst = ds_meter
        else:
            ds_src_sens_lst = [ds_meter, ds_meter]

        if plotobj is None and if_plot:
            plotobj = DataManipulator(1)

        mea_dict = self.get_measure_dict(
            (
                "V_source_fixed_ac",
                "V_source_fixed_dc",
                "V_sense_ac",
                "I_sense_dc",
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
            special_name=f"{resistor}Ohm-{folder_name}",
            measure_nickname="rt-lockin",
            vary_loop=vary_loop,
            wait_before_vary=wait_before_vary,
            no_start_vary=no_start_vary,
        )

        logger.info("filepath: %s", mea_dict["file_path"])
        logger.info("no of columns(with time column): %d", mea_dict["record_num"])
        logger.info("vary modules: %s", mea_dict["vary_mod"])

        vary_lst, _, _, _ = self._extract_vary(mea_dict)
        # modify the plot configuration
        begin_vary = False
        if if_plot:
            plotobj.live_plot_init(
                2, 2, 2, titles=[["T R", r"T $V_{lockin}$"], [r"T $I_{g}$", "t T"]]
            )
            plotobj.start_saving(mea_dict["plot_record_path"], saving_interval)

        for gen_i in mea_dict["gen_lst"]:
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], gen_i)
            time.sleep(step_time)
            if plotobj is None:
                continue
            plotobj.live_plot_update(
                [0, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 0, 1, 0],
                [gen_i[9], gen_i[9], gen_i[9], gen_i[9]],
                [gen_i[5] / (gen_i[1] / resistor), gen_i[3], gen_i[4], gen_i[7]],
                incremental=True,
            )
            plotobj.live_plot_update(1, 1, 0, gen_i[0], gen_i[9], incremental=True)

            if not begin_vary:
                for funci in vary_lst:
                    funci()
                begin_vary = True

        if if_plot:
            plotobj.stop_saving()

    def b2_measure_VV_V1wI_BTvary_rt_lockin(
        self,
        *,
        resistor: float | str,
        vds: float,
        ds_high: int | str,
        ds_low: int | str,
        ds_meter: Meter | list[Meter],
        ds_compliance: float | str,
        freq: float,
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
        if_plot: bool = True,
        saving_interval: float = 7,
        no_start_vary: bool = True,
        plotobj: Optional[DataManipulator] = None,
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
            if_plot: bool, the individual plot
            saving_interval: float, the saving interval in seconds
        """
        resistor = convert_unit(resistor, "Ohm")[0]
        if isinstance(ds_meter, list):
            logger.validate(len(ds_meter) == 2, "ds_meter must be a list of two meters")
            ds_src_sens_lst = ds_meter
        else:
            ds_src_sens_lst = [ds_meter, ds_meter]

        if plotobj is None and if_plot:
            plotobj = DataManipulator(1)

        mea_dict = self.get_measure_dict(
            (
                "V_source_fixed_ac",
                "V_source_fixed_dc",
                "V_sense_ac",
                "I_sense_dc",
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
            special_name=f"{resistor}Ohm-{folder_name}",
            measure_nickname="rt-lockin",
            vary_loop=vary_loop,
            wait_before_vary=wait_before_vary,
            no_start_vary=no_start_vary,
            manual_record_columns=["time", "V_source", "V_source2", "X", "Y", "R", "Theta", "I", "B", "T", "TB", "TC", "TD"]
        )

        logger.info("filepath: %s", mea_dict["file_path"])
        logger.info("no of columns(with time column): %d", mea_dict["record_num"])
        logger.info("vary modules: %s", mea_dict["vary_mod"])

        vary_lst, _, _, _ = self._extract_vary(mea_dict)
        # modify the plot configuration
        begin_vary = False
        if if_plot:
            plotobj.live_plot_init(
                2, 2, 4, titles=[["T R", r"T $V_{lockin}$"], [r"T $I_{g}$", "t T"]]
            )
            plotobj.start_saving(mea_dict["plot_record_path"], saving_interval)

        for gen_i in mea_dict["gen_lst"]:
            gen_i = list(gen_i) + [self.instrs["itc"].ls.B.temperature, self.instrs["itc"].ls.C.temperature, self.instrs["itc"].ls.D.temperature]
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], gen_i)
            time.sleep(step_time)
            if plotobj is None:
                continue
            plotobj.live_plot_update(
                [0, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 0, 1, 0],
                [gen_i[9], gen_i[9], gen_i[9], gen_i[9]],
                [gen_i[5] / (gen_i[1] / resistor), gen_i[3], gen_i[4], gen_i[7]],
                incremental=True,
            )
            plotobj.live_plot_update(
                [1, 1, 1, 1], 
                [1, 1, 1, 1],
                [0, 1, 2, 3],
                [gen_i[0], gen_i[0], gen_i[0], gen_i[0]],
                [gen_i[9], gen_i[10], gen_i[11], gen_i[12]],
                incremental=True
                )

            if not begin_vary:
                for funci in vary_lst:
                    funci()
                begin_vary = True

        if if_plot:
            plotobj.stop_saving()

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
        folder_name: str = "",
        step_time: float = 0.5,
        if_plot: bool = True,
        saving_interval: float = 7,
        plotobj: DataManipulator = None,
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
            if_plot: bool, the individual plot
            saving_interval: float, the saving interval in seconds
        """
        if isinstance(ds_meter, list):
            logger.validate(len(ds_meter) == 2, "ds_meter must be a list of two meters")
            ds_src_sens_lst = ds_meter
        else:
            ds_src_sens_lst = [ds_meter, ds_meter]

        if plotobj is None and if_plot:
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
                    "I_sense_dc",
                    "I_sense_dc",
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
                    "I_sense_ac",
                    "I_sense_dc",
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
        if if_plot:
            plotobj.live_plot_init(
                1, 2, 1 if freq is None else 2, titles=[[r"$R V_g$", r"$I_g V_g$"]]
            )
            plotobj.start_saving(mea_dict["plot_record_path"], saving_interval)

        for i in mea_dict["gen_lst"]:
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], i)
            time.sleep(step_time)
            if plotobj is None:
                continue
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

        if if_plot:
            plotobj.stop_saving()

    def measure_VVswp_V1wI_BT_gateswp_lockin(
        self,
        *,
        vds: float,
        ds_high: int | str,
        ds_low: int | str,
        ds_meter: Meter | list[Meter],
        ds_compliance: float | str,
        freq: float,
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
        if_plot: bool = True,
        saving_interval: float = 7,
        resistor: float | str,
        plotobj: DataManipulator = None,
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
            if_plot: bool, the individual plot
            saving_interval: float, the saving interval in seconds
            resistor: float | str, the resistor for the approximate curr source
        """
        resistor = convert_unit(resistor, "Ohm")[0]
        if isinstance(ds_meter, list):
            logger.validate(len(ds_meter) == 2, "ds_meter must be a list of two meters")
            ds_src_sens_lst = ds_meter
        else:
            ds_src_sens_lst = [ds_meter, ds_meter]

        if plotobj is None and if_plot:
            plotobj = DataManipulator(1)
        if vg_swp_lst is not None:
            vg_swpmode = "manual"
            swp_lst = [vg_swp_lst]
        else:
            swp_lst = None

        mea_dict = self.get_measure_dict(
            (
                "V_source_fixed_ac",
                "V_source_sweep_dc",
                "V_sense_ac",
                "I_sense_dc",
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
            special_name=f"{resistor}Ohm-{folder_name}",
            sweep_tables=swp_lst,
            measure_nickname="swp-gate-lockin",
        )

        logger.info("filepath: %s", mea_dict["file_path"])
        logger.info("no of columns(with time column): %d", mea_dict["record_num"])
        logger.info("vary modules: %s", mea_dict["vary_mod"])

        # modify the plot configuration
        # note i[0] is timer
        if if_plot:
            plotobj.live_plot_init(
                1, 3, 1, titles=[[r"$R V_g$", r"$I_g V_g$", r"$\Theta\  V_g$"]]
            )
            plotobj.start_saving(mea_dict["plot_record_path"], saving_interval)

        for i in mea_dict["gen_lst"]:
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], i)
            time.sleep(step_time)
            if plotobj is None:
                continue
            plotobj.live_plot_update(
                [0, 0, 0],
                [0, 2, 1],
                [0, 0, 0],
                [i[2], i[2], i[2]],
                [i[3] / (vds / resistor), i[6], i[7]],
                incremental=True,
            )

        if if_plot:
            plotobj.stop_saving()

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
        folder_name: str = "",
        step_time: float = 0.3,
        if_plot: bool = True,
        saving_interval: float = 7,
        plotobj: DataManipulator = None,
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
            if_plot: bool, the individual plot
            saving_interval: float, the saving interval in seconds
        """
        if isinstance(ds_meter, list):
            logger.validate(len(ds_meter) == 2, "ds_meter must be a list of two meters")
            ds_src_sens_lst = ds_meter
        else:
            ds_src_sens_lst = [ds_meter, ds_meter]
        if plotobj is None and if_plot:
            plotobj = DataManipulator(1)
        if vds_swp_lst is not None:
            vds_swpmode = "manual"

        if freq is None:
            mea_dict = self.get_measure_dict(
                (
                    "V_source_sweep_dc",
                    "V_source_fixed_dc",
                    "I_sense_dc",
                    "I_sense_dc",
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
                    "I_sense_ac",
                    "I_sense_dc",
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
        if if_plot:
            plotobj.live_plot_init(
                1, 2, 1 if freq is None else 2, titles=[[r"V_{ds} I", r"V_{ds} T"]]
            )
            plotobj.start_saving(mea_dict["plot_record_path"], saving_interval)

        for i in mea_dict["gen_lst"]:
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], i)
            time.sleep(step_time)
            if plotobj is None:
                continue
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

        if if_plot:
            plotobj.stop_saving()

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
        folder_name: str = "",
        step_time: float = 0.3,
        wait_before_vary: float = 5,
        vary_loop: bool = True,
        if_plot: bool = True,
        saving_interval: float = 7,
        plotobj: Optional[DataManipulator] = None,
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
            if_plot: bool, the individual plot
            saving_interval: float, the saving interval in seconds
            plotobj: DataManipulator, the plotting object
        """
        if isinstance(ds_meter, list):
            logger.validate(len(ds_meter) == 2, "ds_meter must be a list of two meters")
            ds_src_sens_lst = ds_meter
        else:
            ds_src_sens_lst = [ds_meter, ds_meter]

        if plotobj is None and if_plot:
            plotobj = DataManipulator(1)
        if freq is None:
            mea_dict = self.get_measure_dict(
                (
                    "V_source_fixed_dc",
                    "V_source_fixed_dc",
                    "I_sense_dc",
                    "I_sense_dc",
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
                    "I_sense_ac",
                    "I_sense_dc",
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

        vary_lst, _, _, _ = self._extract_vary(mea_dict)
        # modify the plot configuration
        begin_vary = False
        if if_plot:
            plotobj.live_plot_init(
                1, 3, 1 if freq is None else 2, titles=[["B I", "B T", "t B"]]
            )
            plotobj.start_saving(mea_dict["plot_record_path"], saving_interval)

        for gen_i in mea_dict["gen_lst"]:
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], gen_i)
            time.sleep(step_time)
            if plotobj is None:
                continue
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

            if not begin_vary:
                for funci in vary_lst:
                    funci()
                begin_vary = True

        if if_plot:
            plotobj.stop_saving()

#    def measure_VV_VwV2wI_BvaryT_rhloop_lockin(
#        self,
#        *,
#        vds: float,
#        ds_high: int | str,
#        ds_low: int | str,
#        ds_meter_source: Meter,
#        ds_meter_1w: Meter,
#        ds_meter_2w: Meter,
#        ds_compliance: float | str,
#        freq: float,
#        vg: float,
#        vg_high: int | str,
#        vg_meter: Meter,
#        vg_compliance: float | str,
#        field_start: float,
#        field_end: float,
#        temperature: float,
#        folder_name: str = "",
#        step_time: float = 0.3,
#        wait_before_vary: float = 5,
#        vary_loop: bool = True,
#        if_plot: bool = True,
#        saving_interval: float = 7,
#        plotobj: DataManipulator = None,
#    ):
#        """
#        measure the V-V and I-I curve using one or two source meters, with other info (B, T, etc.)
#
#        Args:
#            vds: float, the drain-source voltage
#            ds_high: int | str, the high terminal of the drain-source
#            ds_low: int | str, the low terminal of the drain-source
#            ds_meter_source: Meter, the meter used for source
#            ds_meter_1w: Meter, the meter used for the 1w signal
#            ds_meter_2w: Meter, the meter used for the 2w signal
#            ds_compliance: float | str, the compliance of the drain-source meter
#            freq: float, the frequency
#            vg: float, the gate voltage
#            vg_high: int | str, the high terminal of the gate
#            vg_meter: Meter, the meter used for the gate
#            vg_compliance: float | str, the compliance of the gate meter
#            field_start: float, the start field
#            field_end: float, the end field
#            temperature: float, the temperature
#            folder_name: str, the folder name
#            step_time: float, the step time
#            wait_before_vary: float, the wait before vary
#            vary_loop: bool, the vary loop
#            if_plot: bool, the individual plot
#            saving_interval: float, the saving interval in seconds
#        """
#        if plotobj is None and if_plot:
#            plotobj = DataManipulator(1)
#        mea_dict = self.get_measure_dict(
#            (
#                "V_source_fixed_ac",
#                "V_source_fixed_dc",
#                "V_sense_ac",
#                "V_sense_ac",
#                "I_sense_dc",
#                "B_vary",
#                "T_fixed",
#            ),
#            vds,
#            freq,
#            ds_high,
#            ds_low,
#            vg,
#            vg_high,
#            0,
#            "1w",
#            ds_high,
#            ds_low,
#            "2w",
#            ds_high,
#            ds_low,
#            "gate",
#            vg_high,
#            0,
#            field_start,
#            field_end,
#            temperature,
#            wrapper_lst=[
#                ds_meter_source,
#                vg_meter,
#                ds_meter_1w,
#                ds_meter_2w,
#                vg_meter,
#            ],
#            compliance_lst=[ds_compliance, vg_compliance],
#            if_combine_gen=True,  # False for coexistence of vary and mapping
#            special_name=folder_name,
#            measure_nickname="rh-loop-lockin",
#            vary_loop=vary_loop,
#            wait_before_vary=wait_before_vary,
#        )
#
#        logger.info("filepath: %s", mea_dict["file_path"])
#        logger.info("no of columns(with time column): %d", mea_dict["record_num"])
#        logger.info("vary modules: %s", mea_dict["vary_mod"])
#
#        ds_meter_1w.reference_set(harmonic=1)
#        ds_meter_2w.reference_set(harmonic=2)
#
#        vary_lst, _, _, _ = self._extract_vary(mea_dict)
#        # modify the plot configuration
#        begin_vary = False
#        if if_plot:
#            plotobj.live_plot_init(2, 2, 2, titles=[["B I1w", "B I2w"], ["B T", "t B"]])
#            plotobj.start_saving(mea_dict["plot_record_path"], saving_interval)
#
#        for gen_i in mea_dict["gen_lst"]:
#            self.record_update(mea_dict["file_path"], mea_dict["record_num"], gen_i)
#            time.sleep(step_time)
#            if plotobj is not None:
#                plotobj.live_plot_update(
#                    [0, 0, 0, 0, 1],
#                    [0, 0, 1, 1, 0],
#                    [0, 1, 0, 1, 0],
#                    [gen_i[12], gen_i[12], gen_i[12], gen_i[12], gen_i[12]],
#                    [gen_i[3], gen_i[4], gen_i[7], gen_i[8], gen_i[13]],
#                    incremental=True,
#                )
#                plotobj.live_plot_update(1, 1, 0, gen_i[0], gen_i[12], incremental=True)
#
#            if not begin_vary:
#                for funci in vary_lst:
#                    funci()
#                begin_vary = True
#
#        if if_plot:
#            plotobj.stop_saving()

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
        folder_name: str = "",
        step_time: float = 1,
        if_plot: bool = True,
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
            if_plot=if_plot,
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
        folder_name: str = "",
        step_time: float = 1,
        if_plot: bool = True,
        ds_gate_order: tuple[int, int] = (0, 1),
        calculate_from_ds: Optional[Callable] = None,
        contour_ac: Literal["X", "Y", "R", "Theta"] = "R",
        saving_interval: float = 7,
        plotobj: DataManipulator = None,
    ):
        """
        NOTE: the sweep order is important, for order (0,1), it means the inner loop is vg, the outer loop is vds (if not constrained)

        Args:
            vds_max: float, the drain-source voltage max
            ds_map_lst: Sequence[float], the drain-source voltage map list
            ds_high: int | str, the drain-source high
            ds_low: int | str, the drain-source low
            ds_map_lst: Sequence[float], the drain-source voltage map list
            freq: Optional[float], the frequency
            ds_meter: Meter | list[Meter], the drain-source meter
            ds_compliance: float | str, the compliance of the drain-source meter
            vg: float, the gate voltage
            gate_map_lst: Sequence[float], the gate voltage map list
            vg_high: int | str, the gate high
            vg_meter: Meter, the gate meter
            vg_compliance: float | str, the compliance of the gate meter
            field: float, the field
            temperature: float, the temperature
            folder_name: str, the folder name
            step_time: float, the step time
            if_plot: bool, the individual plot
            ds_gate_order: tuple[int, int], the drain-source gate order
            calculate_from_ds: Callable, the function to calculate the gate voltage from the drain-source voltage
            constrained: bool, the constrained
            folder_name: str, the folder name
            step_time: float, the step time
            if_plot: bool, the individual plot
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

        if plotobj is None and if_plot:
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
                    "I_sense_dc",
                    "I_sense_dc",
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
                    "I_sense_ac",
                    "I_sense_dc",
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
        if if_plot:
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
            time.sleep(step_time)
            if plotobj is None:
                continue
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

        if if_plot:
            plotobj.stop_saving()

    def measure_VVswpVswp_III_BT_dualgatemapping(
        self,
        *,
        constrained: bool = False,
        vds: float,
        freq: Optional[float] = None,
        ds_high: int | str,
        ds_low: int | str,
        ds_meter: Meter | list[Meter],
        ds_compliance: float | str,
        vg1_max: float,
        vg1_map_lst: Sequence[float],
        vg1_high: int | str,
        vg1_meter: Meter,
        vg1_compliance: float | str,
        vg2_max: float,
        vg2_map_lst: Sequence[float],
        vg2_high: int | str,
        vg2_meter: Meter,
        vg2_compliance: float | str,
        field: float,
        temperature: float,
        folder_name: str = "",
        step_time: float = 1,
        vg1_vg2_order: tuple[int, int] = (0, 1),
        calculate_from_vg1: Optional[Callable] = None,
        contour_ac: Literal["X", "Y", "R", "Theta"] = "R",
        saving_interval: float = 7,
        plotobj: DataManipulator = None,
        if_plot: bool = True,
    ):
        """
        NOTE: the sweep order is important, for order (0,1), it means the inner loop is vg2, the outer loop is vg1 (if not constrained)

        Args:
            vds: float, the drain-source voltage
            freq: float, the frequency (if ac)
            ds_high: int | str, the high terminal of the drain-source
            ds_low: int | str, the low terminal of the drain-source
            ds_meter: Meter | list[Meter], the meter used for both source and sense or two meters separately in a list
            ds_compliance: float | str, the compliance of the drain-source meter
            vg1_max: float, the maximum gate voltage 1
            vg1_step: float, the step gate voltage 1
            vg1_high: int | str, the high terminal of the gate 1
            vg1_map_lst: Sequence[float], the drain-source voltage map list
            vg1_meter: Meter | list[Meter], the drain-source meter
            vg1_compliance: float | str, the compliance of the drain-source meter
            vg2_max: float, the maximum gate voltage
            vg2_step: float, the step gate voltage
            vg2_high: int | str, the gate high
            vg2_map_lst: Sequence[float], the gate voltage map list
            vg2_meter: Meter, the gate meter
            vg2_compliance: float | str, the compliance of the gate meter
            field: float, the field
            temperature: float, the temperature
            folder_name: str, the folder name
            step_time: float, the step time
            vg1_vg2_order: tuple[int, int], the drain-source gate order
            calculate_from_vg1: Callable, the function to calculate the property from the vg1 voltage
            constrained: bool, the constrained
            contour_ac: Literal["X", "Y", "R", "Theta"], the ac sense property used for contour plot
            saving_interval: float, the saving interval in seconds
            plotobj: DataManipulator, the plot object
            if_plot: bool, the if plot
        """
        contour_ac_idx = {"X": 4, "Y": 5, "R": 6, "Theta": 7}[contour_ac]
        if isinstance(ds_meter, list):
            logger.validate(len(ds_meter) == 2, "ds_meter must be a list of two meters")
            ds_src_sens_lst = ds_meter
        else:
            ds_src_sens_lst = [ds_meter, ds_meter]

        if plotobj is None and if_plot:
            plotobj = DataManipulator(1)
        if not constrained:
            map_lsts = self.create_mapping(vg1_map_lst, vg2_map_lst, idxs=vg1_vg2_order)
            # =======only applied to ds and vg mapping=========
            if_inner_loop_is_vg1 = vg1_vg2_order[0] == 1
            inner_loop_len = (
                len(vg1_map_lst) if if_inner_loop_is_vg1 else len(vg2_map_lst)
            )
            # ==================================================
            if calculate_from_vg1 is not None:
                logger.warning("calculate_fromds causes no effect when not constrained")
        else:
            if calculate_from_vg1 is None:
                calculate_from_vg1 = lambda x: x
            map_lsts = [vg1_map_lst, vg2_map_lst]

        # Core configuration
        # Generate the measurement generator
        if freq is None:
            mea_dict = self.get_measure_dict(
                (
                    "V_source_fixed_dc",
                    "V_source_sweep_dc",
                    "V_source_sweep_dc",
                    "I_sense_dc",
                    "I_sense_dc",
                    "I_sense_dc",
                    "B_fixed",
                    "T_fixed",
                ),
                vds,
                ds_high,
                ds_low,
                vg1_max,
                0,
                vg1_high,
                0,
                "manual",
                vg2_max,
                0,
                vg2_high,
                0,
                "manual",
                "",
                ds_high,
                ds_low,
                "",
                vg1_high,
                0,
                "",
                vg2_high,
                0,
                field,
                temperature,
                wrapper_lst=[
                    ds_src_sens_lst[0],
                    vg1_meter,
                    vg2_meter,
                    ds_src_sens_lst[1],
                    vg1_meter,
                    vg2_meter,
                ],
                compliance_lst=[ds_compliance, vg1_compliance, vg2_compliance],
                sr830_current_resistor=None,  # only float
                if_combine_gen=True,  # False for coexistence of vary and mapping
                special_name=folder_name,
                sweep_tables=map_lsts,
                measure_nickname="dualgate-mapping-dc",
            )
        else:
            mea_dict = self.get_measure_dict(
                (
                    "V_source_fixed_ac",
                    "V_source_sweep_dc",
                    "V_source_sweep_dc",
                    "I_sense_ac",
                    "I_sense_dc",
                    "I_sense_dc",
                    "B_fixed",
                    "T_fixed",
                ),
                vds,
                freq,
                ds_high,
                ds_low,
                vg1_max,
                0,
                vg1_high,
                0,
                "manual",
                vg2_max,
                0,
                vg2_high,
                0,
                "manual",
                "",
                ds_high,
                ds_low,
                "",
                vg1_high,
                0,
                "",
                vg2_high,
                0,
                field,
                temperature,
                wrapper_lst=[
                    ds_src_sens_lst[0],
                    vg1_meter,
                    vg2_meter,
                    ds_src_sens_lst[1],
                    vg1_meter,
                    vg2_meter,
                ],
                compliance_lst=[ds_compliance, vg1_compliance, vg2_compliance],
                sr830_current_resistor=None,  # only float
                if_combine_gen=True,  # False for coexistence of vary and mapping
                special_name=folder_name,
                sweep_tables=map_lsts,
                measure_nickname="dualgate-mapping-ac",
            )

        logger.info("filepath: %s", mea_dict["file_path"])
        logger.info("no of columns(with time column): %d", mea_dict["record_num"])
        logger.info("vary modules: %s", mea_dict["vary_mod"])
        # modify the plot configuration
        if plotobj is not None:
            if not constrained:
                titles = (
                    [[r"$V_{g1}-I_{ds}$"], [r"$V_{g1}-I_{g}s$"], ["contour"]]
                    if if_inner_loop_is_vg1
                    else [[r"$V_{g2} I_{ds}$"], [r"$V_{g2} I_{g}s$"], ["contour"]]
                )
                axes_labels = (
                    [
                        [[r"$V_{g1}$", r"$I_{ds}$"]],
                        [[r"$V_{g1}$", r"$I_{g}$"]],
                        [[r"$V_{g1}$", r"$V_{g2}$"]],
                    ]
                    if if_inner_loop_is_vg1
                    else [
                        [[r"$V_{g2}$", r"$I_{ds}$"]],
                        [[r"$V_{g2}$", r"$I_{g}$"]],
                        [[r"$V_{g1}$", r"$V_{g2}$"]],
                    ]
                )
                plotobj.live_plot_init(
                    3,
                    1,
                    2,
                    plot_types=[["scatter"], ["scatter"], ["contour"]],
                    titles=titles,
                    axes_labels=axes_labels,
                )
            else:
                plotobj.live_plot_init(
                    3,
                    1,
                    1,
                    plot_types=[["scatter"], ["scatter"], ["scatter"]],
                    titles=[["n I_{ds}"], ["V_{g1} I_{g1}"], ["V_{g2} I_{g2}"]],
                    axes_labels=[
                        [[r"$n$", r"$I_{ds}$"]],
                        [[r"$V_{g1}$", r"$I_{g1}$"]],
                        [[r"$V_{g2}$", r"$I_{g2}$"]],
                    ],
                )

            plotobj.start_saving(mea_dict["plot_record_path"], saving_interval)

        for i in mea_dict["gen_lst"]:
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], i)
            if plotobj is None:
                continue
            if not constrained:
                x_data = i[2] if if_inner_loop_is_vg1 else i[3]
                if freq is None:
                    plotobj.live_plot_update(
                        [0, 1, 1],
                        [0, 0, 0],
                        [0, 0, 1],
                        [x_data, x_data, x_data],
                        [i[4], i[5], i[6]],
                        incremental=True,
                        max_points=inner_loop_len,
                    )
                    plotobj.live_plot_update(
                        2,
                        0,
                        0,
                        i[2],
                        i[3],
                        i[4],
                        incremental=True,
                    )
                else:
                    plotobj.live_plot_update(
                        [0, 0, 1],
                        [0, 0, 0],
                        [0, 1, 0],
                        [x_data, x_data, x_data],
                        [i[6], i[8], i[9]],
                        incremental=True,
                        max_points=inner_loop_len,
                    )
                    plotobj.live_plot_update(
                        2,
                        0,
                        0,
                        i[2],
                        i[3],
                        i[contour_ac_idx],
                        incremental=True,
                    )
            else:
                if freq is None:
                    plotobj.live_plot_update(
                        [0, 1, 2],
                        [0, 0, 0],
                        [0, 0, 0],
                        [calculate_from_vg1(i[2]), i[2], i[3]],
                        [i[4], i[5], i[6]],
                        incremental=True,
                    )
                else:
                    plotobj.live_plot_update(
                        [0, 1, 2],
                        [0, 0, 0],
                        [0, 0, 0],
                        [calculate_from_vg1(i[2]), i[2], i[3]],
                        [i[6], i[8], i[9]],
                        incremental=True,
                    )
            time.sleep(step_time)

        if plotobj is not None:
            plotobj.stop_saving()

    def measure_VVswpVswp_VII_BT_dualgatemapping_lockin(
        self,
        *,
        constrained: bool = False,
        resistor: float | str,
        vds: float,
        freq: float,
        ds_high: int | str,
        ds_low: int | str,
        ds_meter: Meter | list[Meter],
        ds_compliance: float | str,
        vg1_max: float,
        vg1_map_lst: Sequence[float],
        vg1_high: int | str,
        vg1_meter: Meter,
        vg1_compliance: float | str,
        vg2_max: float,
        vg2_map_lst: Sequence[float],
        vg2_high: int | str,
        vg2_meter: Meter,
        vg2_compliance: float | str,
        field: float,
        temperature: float,
        folder_name: str = "",
        step_time: float = 1,
        vg1_vg2_order: tuple[int, int] = (0, 1),
        calculate_from_vg1: Optional[Callable] = None,
        contour_ac: Literal["X", "Y", "R", "Theta"] = "R",
        saving_interval: float = 7,
        plotobj: DataManipulator = None,
        if_plot: bool = True,
    ):
        """
        NOTE: the sweep order is important, for order (0,1), it means the inner loop is vg2, the outer loop is vg1 (if not constrained)

        Args:
            vds: float, the drain-source voltage
            freq: float, the frequency (if ac)
            ds_high: int | str, the high terminal of the drain-source
            ds_low: int | str, the low terminal of the drain-source
            ds_meter: Meter | list[Meter], the meter used for both source and sense or two meters separately in a list
            ds_compliance: float | str, the compliance of the drain-source meter
            vg1_max: float, the maximum gate voltage 1
            vg1_step: float, the step gate voltage 1
            vg1_high: int | str, the high terminal of the gate 1
            vg1_map_lst: Sequence[float], the drain-source voltage map list
            vg1_meter: Meter | list[Meter], the drain-source meter
            vg1_compliance: float | str, the compliance of the drain-source meter
            vg2_max: float, the maximum gate voltage
            vg2_step: float, the step gate voltage
            vg2_high: int | str, the gate high
            vg2_map_lst: Sequence[float], the gate voltage map list
            vg2_meter: Meter, the gate meter
            vg2_compliance: float | str, the compliance of the gate meter
            field: float, the field
            temperature: float, the temperature
            folder_name: str, the folder name
            step_time: float, the step time
            vg1_vg2_order: tuple[int, int], the drain-source gate order
            calculate_from_vg1: Callable, the function to calculate the property from the vg1 voltage
            constrained: bool, the constrained
            contour_ac: Literal["X", "Y", "R", "Theta"], the ac sense property used for contour plot
            saving_interval: float, the saving interval in seconds
            plotobj: DataManipulator, the plot object
            if_plot: bool, the if plot
        """
        contour_ac_idx = {"X": 4, "Y": 5, "R": 6, "Theta": 7}[contour_ac]
        if isinstance(ds_meter, list):
            logger.validate(len(ds_meter) == 2, "ds_meter must be a list of two meters")
            ds_src_sens_lst = ds_meter
        else:
            ds_src_sens_lst = [ds_meter, ds_meter]

        if plotobj is None and if_plot:
            plotobj = DataManipulator(1)
        if not constrained:
            map_lsts = self.create_mapping(vg1_map_lst, vg2_map_lst, idxs=vg1_vg2_order)
            # =======only applied to ds and vg mapping=========
            if_inner_loop_is_vg1 = vg1_vg2_order[0] == 1
            inner_loop_len = (
                len(vg1_map_lst) if if_inner_loop_is_vg1 else len(vg2_map_lst)
            )
            # ==================================================
            if calculate_from_vg1 is not None:
                logger.warning("calculate_fromds causes no effect when not constrained")
        else:
            if calculate_from_vg1 is None:
                calculate_from_vg1 = lambda x: x
            map_lsts = [vg1_map_lst, vg2_map_lst]

        # Core configuration
        # Generate the measurement generator
        mea_dict = self.get_measure_dict(
            (
                "V_source_fixed_ac",
                "V_source_sweep_dc",
                "V_source_sweep_dc",
                "V_sense_ac",
                "I_sense_dc",
                "I_sense_dc",
                "B_fixed",
                "T_fixed",
            ),
            vds,
            freq,
            ds_high,
            ds_low,
            vg1_max,
            0,
            vg1_high,
            0,
            "manual",
            vg2_max,
            0,
            vg2_high,
            0,
            "manual",
            "",
            ds_high,
            ds_low,
            "",
            vg1_high,
            0,
            "",
            vg2_high,
            0,
            field,
            temperature,
            wrapper_lst=[
                ds_src_sens_lst[0],
                vg1_meter,
                vg2_meter,
                ds_src_sens_lst[1],
                vg1_meter,
                vg2_meter,
            ],
            compliance_lst=[ds_compliance, vg1_compliance, vg2_compliance],
            sr830_current_resistor=None,  # only float
            if_combine_gen=True,  # False for coexistence of vary and mapping
            special_name=f"{resistor}Ohm-{folder_name}",
            sweep_tables=map_lsts,
            measure_nickname="dualgate-mapping-lockin",
        )

        resistor = convert_unit(resistor, "Ohm")[0]
        logger.info("filepath: %s", mea_dict["file_path"])
        logger.info("no of columns(with time column): %d", mea_dict["record_num"])
        logger.info("vary modules: %s", mea_dict["vary_mod"])
        # modify the plot configuration
        if plotobj is not None:
            if not constrained:
                titles = (
                    [
                        [r"$V_{g1}-R$", r"$V_{g1}-V_{lockin}$"],
                        [r"$V_{g1}-I_{g}s$", "contour"],
                    ]
                    if if_inner_loop_is_vg1
                    else [
                        [r"$V_{g2}-R$", r"$V_{g2}-V_{lockin}$"],
                        [r"$V_{g2}-I_{g}s$", "contour"],
                    ]
                )
                axes_labels = (
                    [
                        [[r"$V_{g1}$", r"R"], [r"$V_{g1}$", r"$V_{lockin}$"]],
                        [[r"$V_{g1}$", r"$I_{g}$"], [r"$V_{g1}$", r"$V_{g2}$"]],
                    ]
                    if if_inner_loop_is_vg1
                    else [
                        [[r"$V_{g2}$", r"R"], [r"$V_{g2}$", r"$V_{lockin}$"]],
                        [[r"$V_{g2}$", r"$I_{g}$"], [r"$V_{g1}$", r"$V_{g2}$"]],
                    ]
                )
                plotobj.live_plot_init(
                    2,
                    2,
                    2,
                    plot_types=[["scatter", "scatter"], ["scatter", "contour"]],
                    titles=titles,
                    axes_labels=axes_labels,
                )
            else:
                plotobj.live_plot_init(
                    2,
                    2,
                    2,
                    plot_types=[["scatter", "scatter"], ["scatter", "scatter"]],
                    titles=[
                        [r"$n-R$", r"$n-V_{lockin}$"],
                        [r"$V_{g1}-I_{g1}$", r"$V_{g2} I_{g2}$"],
                    ],
                    axes_labels=[
                        [[r"$n$", r"$R$"], [r"$n$", r"$V_{lockin}$"]],
                        [[r"$V_{g1}$", r"$I_{g1}$"], [r"$V_{g2}$", r"$I_{g2}$"]],
                    ],
                )

            plotobj.start_saving(mea_dict["plot_record_path"], saving_interval)

        for i in mea_dict["gen_lst"]:
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], i)
            if plotobj is None:
                continue
            if not constrained:
                x_data = i[2] if if_inner_loop_is_vg1 else i[3]
                plotobj.live_plot_update(
                    [0, 0, 0, 1, 1],
                    [0, 1, 1, 0, 0],
                    [0, 0, 1, 0, 1],
                    [x_data, x_data, x_data, x_data, x_data],
                    [i[6] / (i[1] / resistor), i[4], i[5], i[8], i[9]],
                    incremental=True,
                    max_points=inner_loop_len,
                )
                plotobj.live_plot_update(
                    1,
                    1,
                    0,
                    i[2],
                    i[3],
                    i[contour_ac_idx],
                    incremental=True,
                )
            else:
                prop_n = calculate_from_vg1(i[2])
                plotobj.live_plot_update(
                    [0, 0, 0, 1, 1],
                    [0, 1, 1, 0, 1],
                    [0, 0, 1, 0, 0],
                    [prop_n, prop_n, prop_n, i[2], i[3]],
                    [i[6] / (i[1] / resistor), i[4], i[5], i[8], i[9]],
                    incremental=True,
                )
            time.sleep(step_time)

        if plotobj is not None:
            plotobj.stop_saving()

    def _extract_vary(
        self, mea_dict: dict
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


"""
#    def measure_VswpVswp_II_BvaryT_dsgatemapping_rhloop(
#        self,
#        *,
#        constrained: bool = False,
#        vds_max: float,
#        freq: Optional[float] = None,
#        ds_map_lst: Sequence[float],
#        ds_high: int | str,
#        ds_low: int | str,
#        ds_meter: Meter | list[Meter],
#        ds_compliance: float | str,
#        vg: float,
#        gate_map_lst: Sequence[float],
#        vg_high: int | str,
#        vg_meter: Meter,
#        vg_compliance: float | str,
#        field_start: float,
#        field_end: float,
#        temperature: float,
#        folder_name: str = "",
#        step_time: float = 1,
#        if_plot: bool = True,
#        ds_gate_order: tuple[int, int] = (0, 1),
#        calculate_from_ds: Optional[Callable] = None,
#        contour_ac: Literal["X", "Y", "R", "Theta"] = "R",
#        saving_interval: float = 7,
#    ):
#        
#        NOTE: the sweep order is important, for order (0,1), it means the inner loop is vg, the outer loop is vds (if not constrained)
#        ##TODO:: not completed (not needed currently), current can't ensure senser generatoe continue to work after varying on each mapping point
#
#        Args:
#            freq: Optional[float], the frequency
#            ds_meter: Meter | list[Meter], the drain-source meter
#            ds_compliance: float | str, the compliance of the drain-source meter
#            vg_meter: Meter, the gate meter
#            vg_compliance: float | str, the compliance of the gate meter
#            field_start: float, the field start
#            field_end: float, the field end
#            temperature: float, the temperature
#            folder_name: str, the folder name
#            step_time: float, the step time
#            if_plot: bool, the individual plot
#            ds_gate_order: tuple[int, int], the drain-source gate order
#            calculate_from_ds: Callable, the function to calculate the gate voltage from the drain-source voltage
#            constrained: bool, the constrained
#            vds_max: float, the drain-source voltage max
#            ds_map_lst: Sequence[float], the drain-source voltage map list
#            ds_high: int | str, the drain-source high
#            ds_low: int | str, the drain-source low
#            vg: float, the gate voltage
#            gate_map_lst: Sequence[float], the gate voltage map list
#            vg_high: int | str, the gate high
#            vg_meter: Meter, the gate meter
#            vg_compliance: float | str, the compliance of the gate meter
#            field: float, the field
#            temperature: float, the temperature
#            folder_name: str, the folder name
#            step_time: float, the step time
#            if_plot: bool, the individual plot
#            ds_gate_order: tuple[int, int], the drain-source gate order
#            calculate_from_ds: Callable, the function to calculate the targeted property from the drain-source voltage
#            contour_ac: Literal["X", "Y", "R", "Theta"], the ac sense property used for contour plot
#            saving_interval: float, the saving interval in seconds
#        
#        contour_ac_idx = {"X": 3, "Y": 4, "R": 5, "Theta": 6}[contour_ac]
#        if isinstance(ds_meter, list):
#            logger.validate(len(ds_meter) == 2, "ds_meter must be a list of two meters")
#            ds_src_sens_lst = ds_meter
#        else:
#            ds_src_sens_lst = [ds_meter, ds_meter]
#
#        plotobj = DataManipulator(1)
#        if not constrained:
#            map_lsts = self.create_mapping(ds_map_lst, gate_map_lst, idxs=ds_gate_order)
#            # =======only applied to ds and vg mapping=========
#            if_inner_loop_is_ds = ds_gate_order[0] == 1
#            inner_loop_len = (
#                len(ds_map_lst) if if_inner_loop_is_ds else len(gate_map_lst)
#            )
#            # ==================================================
#            if calculate_from_ds is not None:
#                logger.warning("calculate_fromds causes no effect when not constrained")
#        else:
#            if calculate_from_ds is None:
#                calculate_from_ds = lambda x: x
#            map_lsts = [ds_map_lst, gate_map_lst]
#
#        # Core configuration
#        # Generate the measurement generator
#        if freq is None:
#            mea_dict = self.get_measure_dict(
#                (
#                    "V_source_sweep_dc",
#                    "V_source_sweep_dc",
#                    "I_sense_dc",
#                    "I_sense_dc",
#                    "B_vary",
#                    "T_fixed",
#                ),
#                vds_max,
#                0,
#                ds_high,
#                ds_low,
#                "manual",
#                vg,
#                0,
#                vg_high,
#                0,
#                "manual",
#                "",
#                ds_high,
#                ds_low,
#                "",
#                vg_high,
#                0,
#                field_start,
#                field_end,
#                temperature,
#                wrapper_lst=[
#                    ds_src_sens_lst[0],
#                    vg_meter,
#                    ds_src_sens_lst[1],
#                    vg_meter,
#                ],
#                compliance_lst=[ds_compliance, vg_compliance],
#                sr830_current_resistor=None,  # only float
#                if_combine_gen=False,  # False for coexistence of vary and mapping
#                special_name=folder_name,
#                sweep_tables=map_lsts,
#                measure_nickname="ds-gate-mapping-rhloop-dc",
#            )
#        else:
#            mea_dict = self.get_measure_dict(
#                (
#                    "V_source_sweep_ac",
#                    "V_source_sweep_dc",
#                    "I_sense_dc",
#                    "I_sense_dc",
#                    "B_vary",
#                    "T_fixed",
#                ),
#                vds_max,
#                0,
#                freq,
#                ds_high,
#                ds_low,
#                "manual",
#                vg,
#                0,
#                vg_high,
#                0,
#                "manual",
#                "",
#                ds_high,
#                ds_low,
#                "",
#                vg_high,
#                0,
#                field_start,
#                field_end,
#                temperature,
#                wrapper_lst=[
#                    ds_src_sens_lst[0],
#                    vg_meter,
#                    ds_src_sens_lst[1],
#                    vg_meter,
#                ],
#                compliance_lst=[ds_compliance, vg_compliance],
#                sr830_current_resistor=None,  # only float
#                if_combine_gen=False,  # False for coexistence of vary and mapping
#                special_name=folder_name,
#                sweep_tables=map_lsts,
#                measure_nickname="ds-gate-mapping-rhloop-ac",
#            )
#
#        logger.info("filepath: %s", mea_dict["file_path"])
#        logger.info("no of columns(with time column): %d", mea_dict["record_num"])
#        logger.info("vary modules: %s", mea_dict["vary_mod"])
#
#        vary_lst, _, _, _ = self._extract_vary(mea_dict)
#        # modify the plot configuration
#        if if_plot:
#            if not constrained:
#                titles = (
#                    [["V_{ds} I_{ds}"], ["V_{ds} I_{g}"], ["contour"]]
#                    if if_inner_loop_is_ds
#                    else [["V_{g} I_{ds}"], ["V_{g} I_{g}"], ["contour"]]
#                )
#                axes_labels = (
#                    [
#                        [[r"$V_{ds}$", r"$I_{ds}$"]],
#                        [[r"$V_{ds}$", r"$I_{g}$"]],
#                        [r"$V_{ds}$", r"$V_{g}$"],
#                    ]
#                    if if_inner_loop_is_ds
#                    else [
#                        [[r"$V_{g}$", r"$I_{ds}$"]],
#                        [[r"$V_{g}$", r"$I_{g}$"]],
#                        [r"$V_{ds}$", r"$V_{g}$"],
#                    ]
#                )
#                plotobj.live_plot_init(
#                    3,
#                    1,
#                    1 if freq is None else 2,
#                    plot_types=[["scatter"], ["scatter"], ["contour"]],
#                    titles=titles,
#                    axes_labels=axes_labels,
#                )
#            else:
#                plotobj.live_plot_init(
#                    2,
#                    1,
#                    1 if freq is None else 2,
#                    plot_types=[["scatter"], ["scatter"]],
#                    titles=[["n I_{ds}"], ["V_{g} I_{g}"]],
#                    axes_labels=[
#                        [[r"$n$", r"$I_{ds}$"]],
#                        [[r"$V_{g}$", r"$I_{ds}$"]],
#                    ],
#                )
#
#        tmp_lst_swp = [[]] * len(mea_dict["swp_idx"])
#        ori_path = mea_dict["plot_record_path"]
#        plot_idx = 0
#        while True:
#            plotobj.start_saving(ori_path.parent / f"{ori_path.stem}-{plot_idx}{ori_path.suffix}", saving_interval)
#            measured_lst = next_lst_gen(mea_dict["gen_lst"])
#            if measured_lst is None:
#                break
#            self.record_update(mea_dict["file_path"], mea_dict["record_num"], measured_lst)
#
#            for n, idx in enumerate(mea_dict["swp_idx"]):
#                tmp_lst_swp[n] = mea_dict["gen_lst"][idx]
#                mea_dict["gen_lst"][idx] = constant_generator(measured_lst[idx])
#
#            if not constrained:
#                x_data = measured_lst[1] if if_inner_loop_is_ds else measured_lst[2]
#                if freq is None:
#                    plotobj.live_plot_update(
#                        [0, 1],
#                        [0] * 2,
#                        [0] * 2,
#                        [x_data, x_data],
#                        [measured_lst[3], measured_lst[4]],
#                        incremental=True,
#                        max_points=inner_loop_len,
#                    )
#                    plotobj.live_plot_update(
#                        2,
#                        0,
#                        0,
#                        i[1],
#                        i[2],
#                        i[3],
#                        incremental=True,
#                    )
#                else:
#                    plotobj.live_plot_update(
#                        [0, 0, 1],
#                        [0] * 3,
#                        [0, 1, 0],
#                        [x_data, x_data, x_data],
#                        [i[3], i[4], i[7]],
#                        incremental=True,
#                        max_points=inner_loop_len,
#                    )
#                    plotobj.live_plot_update(
#                        2,
#                        0,
#                        0,
#                        i[1],
#                        i[2],
#                        i[contour_ac_idx],
#                        incremental=True,
#                    )
#            else:
#                if freq is None:
#                    plotobj.live_plot_update(
#                        [0, 1],
#                        [0] * 2,
#                        [0] * 2,
#                        [calculate_from_ds(i[1]), i[2]],
#                        [i[3], i[4]],
#                        incremental=True,
#                    )
#                else:
#                    plotobj.live_plot_update(
#                        [0, 0, 1],
#                        [0] * 3,
#                        [0, 1, 0],
#                        [calculate_from_ds(i[1]), calculate_from_ds(i[1]), i[2]],
#                        [i[3], i[4], i[7]],
#                        incremental=True,
#                    )
#            time.sleep(step_time)
#        for n, i in enumerate(mea_dict["swp_idx"]):
#            mea_dict["gen_lst"][i] = tmp_lst_swp[n]
#            tmp_lst_swp[n] = []
#
#        if if_plot:
#            plotobj.stop_saving()
"""
