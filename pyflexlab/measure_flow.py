"""
The non-individual plot options have not been written yet
"""

import time
import numpy as np
from typing import Sequence, Callable
from prefect import Flow, task
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
        high: int | str,
        low: int | str,
        swpmode: str,
        meter: Meter,
        compliance: float,
        folder_name: str = None,
        step_time: float = 0.5,
        individual_plot: bool = True,
    ):
        self.measure_Vswp_I_vicurve(
            vmax=vmax,
            vstep=vstep,
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
        high: int | str,
        low: int | str,
        swpmode: str,
        meter: Meter,
        compliance: float,
        folder_name: str = "",
        step_time: float = 0.5,
        individual_plot: bool = True,
    ) -> None:
        """
        measure the V-I curve using ONE DC source meter, no other info (B, T, etc.)

        Args:
            vmax: float, the maximum voltage
            vstep: float, the step voltage
            high: float, the high terminal of the voltage
            low: float, the low terminal of the voltage
            swpmode: str, the sweep mode
            meter: Meter, the meter
            compliance: float, the compliance
            folder_name: str, the folder name
        """
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
            wrapper_lst=[meter, meter],
            compliance_lst=[compliance],
            special_name=folder_name,
            measure_nickname="vi-curve",
        )
        plotobj = DataManipulator(1)

        logger.info("filepath: %s", mea_dict["file_path"])
        logger.info("no of columns(with time column): %d", mea_dict["record_num"])

        if individual_plot:
            plotobj.live_plot_init(
                1,
                1,
                1,
                titles=[[r"$V-I Curve$"]],
                axes_labels=[[[r"$V$", r"$I$"]]],
                line_labels=[[["V-I"]]],
            )
            plotobj.start_saving(mea_dict["plot_record_path"], 7)

        for i in mea_dict["gen_lst"]:
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], i)
            plotobj.live_plot_update(0, 0, 0, i[1], i[2], incremental=True)
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
        ds_meter: Meter,
        ds_compliance: float | str,
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
    ):
        self.measure_VV_II_BTvary_rt(
            vds=vds,
            ds_high=ds_high,
            ds_low=ds_low,
            ds_meter=ds_meter,
            ds_compliance=ds_compliance,
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
        )

    def measure_VV_II_BTvary_rt(
        self,
        *,
        vds: float,
        ds_high: int | str,
        ds_low: int | str,
        ds_meter: Meter,
        ds_compliance: float | str,
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
    ) -> None:
        """
        measure the V-V and I-I curve using TWO DC source meters, with other info (B, T, etc.)
        """

        plotobj = DataManipulator(1)
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
                ds_meter,
                vg_meter,
                ds_meter,
                vg_meter,
            ],
            compliance_lst=[ds_compliance, vg_compliance],
            if_combine_gen=True,  # False for coexistence of vary and mapping
            special_name=folder_name,
            measure_nickname="rt",
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
            plotobj.live_plot_init(1, 3, 1, titles=[["T I_{ds}", "T I_{g}", "t T"]])
            plotobj.start_saving(mea_dict["plot_record_path"], 30)

        for gen_i in mea_dict["gen_lst"]:
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], gen_i)
            plotobj.live_plot_update(
                [0, 0],
                [0, 1],
                [0, 0],
                [gen_i[6], gen_i[6]],
                [gen_i[3], gen_i[4]],
                incremental=True,
            )
            plotobj.live_plot_update(0, 2, 0, gen_i[0], gen_i[6], incremental=True)
            time.sleep(step_time)

            if not begin_vary:
                if counter >= wait_before_vary:
                    for funci in vary_lst:
                        funci()
                    begin_vary = True
                else:
                    counter += step_time

        if individual_plot:
            plotobj.stop_saving()

    @task(name="swp-gate", cache_policy=NO_CACHE)
    def measure_VVswp_II_BT_gateswp_task(
        self,
        *,
        vds: float,
        ds_high: int | str,
        ds_low: int | str,
        ds_meter: Meter,
        ds_compliance: float | str,
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
    ):
        self.measure_VVswp_II_BT_gateswp(
            vds=vds,
            ds_high=ds_high,
            ds_low=ds_low,
            ds_meter=ds_meter,
            ds_compliance=ds_compliance,
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
        )

    def measure_VVswp_II_BT_gateswp(
        self,
        *,
        vds: float,
        ds_high: int | str,
        ds_low: int | str,
        ds_meter: Meter,
        ds_compliance: float | str,
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
    ):
        """
        measure the Vg-I curve using TWO DC source meters, with other info (B, T, etc.) NOTE the vg_swp_lst will override the vg_step, vg_max and vg_swpmode
        """
        plotobj = DataManipulator(1)
        if vg_swp_lst is not None:
            vg_swpmode = "manual"
            swp_lst = [vg_swp_lst]
        else:
            swp_lst = None

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
            wrapper_lst=[ds_meter, vg_meter, ds_meter, vg_meter],
            compliance_lst=[ds_compliance, vg_compliance],
            if_combine_gen=True,  # False for coexistence of vary and mapping
            special_name=folder_name,
            sweep_tables=swp_lst,
            measure_nickname="swp-gate",
        )

        logger.info("filepath: %s", mea_dict["file_path"])
        logger.info("no of columns(with time column): %d", mea_dict["record_num"])
        logger.info("vary modules: %s", mea_dict["vary_mod"])

        # modify the plot configuration
        # note i[0] is timer
        if individual_plot:
            plotobj.live_plot_init(1, 2, 1, titles=[[r"$R V_g$", r"$I_g V_g$"]])
            plotobj.start_saving(mea_dict["plot_record_path"], 30)

        for i in mea_dict["gen_lst"]:
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], i)
            plotobj.live_plot_update(
                [0, 0],
                [0, 1],
                [0, 0],
                [i[2], i[2]],
                [vds / i[3], i[4]],
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
        ds_high: int | str,
        ds_low: int | str,
        vds_swpmode: str,
        vds_swp_lst: Sequence[float] = None,
        ds_meter: Meter,
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
    ):
        self.measure_VswpV_II_BT_vicurve(
            vds_max=vds_max,
            vds_step=vds_step,
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
        )

    def measure_VswpV_II_BT_vicurve(
        self,
        *,
        vds_max: float,
        vds_step: float,
        ds_high: int | str,
        ds_low: int | str,
        vds_swpmode: str,
        vds_swp_lst: Sequence[float] = None,
        ds_meter: Meter,
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
    ):
        """
        measure the Vds-I curve using TWO DC source meters, with other info (B, T, etc.) NOTE the vds_swp_lst will override the vds_step, vds_max and vds_swpmode
        """
        plotobj = DataManipulator(1)
        if vds_swp_lst is not None:
            vds_swpmode = "manual"

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
                ds_meter,
                vg_meter,
                ds_meter,
                vg_meter,
            ],
            compliance_lst=[ds_compliance, vg_compliance],
            if_combine_gen=True,  # False for coexistence of vary and mapping
            special_name=folder_name,
            sweep_tables=[vds_swp_lst],
            measure_nickname="swpds",
        )

        logger.info("filepath: %s", mea_dict["file_path"])
        logger.info("no of columns(with time column): %d", mea_dict["record_num"])
        logger.info("vary modules: %s", mea_dict["vary_mod"])

        # modify the plot configuration
        # note i[0] is timer
        if individual_plot:
            plotobj.live_plot_init(1, 2, 1, titles=[[r"V_{ds} I", r"V_{ds} T"]])
            plotobj.start_saving(mea_dict["plot_record_path"], 30)

        for i in mea_dict["gen_lst"]:
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], i)
            plotobj.live_plot_update(
                [0, 0],
                [0, 1],
                [0, 0],
                [i[1], i[1]],
                [i[3], i[6]],
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
        ds_meter: Meter,
        ds_compliance: float | str,
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
    ):
        self.measure_VV_II_BvaryT_rhloop(
            vds=vds,
            ds_high=ds_high,
            ds_low=ds_low,
            ds_meter=ds_meter,
            ds_compliance=ds_compliance,
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
        )

    def measure_VV_II_BvaryT_rhloop(
        self,
        *,
        vds: float,
        ds_high: int | str,
        ds_low: int | str,
        ds_meter: Meter,
        ds_compliance: float | str,
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
    ):
        plotobj = DataManipulator(1)
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
                ds_meter,
                vg_meter,
                ds_meter,
                vg_meter,
            ],
            compliance_lst=[ds_compliance, vg_compliance],
            if_combine_gen=True,  # False for coexistence of vary and mapping
            special_name=folder_name,
            measure_nickname="rh-loop",
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
            plotobj.live_plot_init(1, 3, 1, titles=[["B I", "B T", "t B"]])
            plotobj.start_saving(mea_dict["plot_record_path"], 30)

        for gen_i in mea_dict["gen_lst"]:
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], gen_i)
            plotobj.live_plot_update(
                [0, 0],
                [0, 1],
                [0, 0],
                [gen_i[5], gen_i[5]],
                [gen_i[3], gen_i[6]],
                incremental=True,
            )
            plotobj.live_plot_update(0, 2, 0, gen_i[0], gen_i[5], incremental=True)
            time.sleep(step_time)

            if not begin_vary:
                if counter >= wait_before_vary:
                    for funci in vary_lst:
                        funci()
                    begin_vary = True
                else:
                    counter += step_time

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
        ds_meter: Meter,
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
    ):
        self.measure_VswpVswp_II_BT_dsgatemapping(
            constrained=constrained,
            vds_max=vds_max,
            ds_map_lst=ds_map_lst,
            ds_high=ds_high,
            ds_low=ds_low,
            ds_meter=ds_meter,
            ds_compliance=ds_compliance,
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
        )

    def measure_VswpVswp_II_BT_dsgatemapping(
        self,
        *,
        constrained: bool = False,
        vds_max: float,
        ds_map_lst: Sequence[float],
        ds_high: int | str,
        ds_low: int | str,
        ds_meter: Meter,
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
        calculate_from_ds: Callable = None,
    ):
        """
        NOTE: the sweep order is important, for order (0,1), it means the inner loop is vg, the outer loop is vds (if not constrained)
        """
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
                ds_meter,
                vg_meter,
                ds_meter,
                vg_meter,
            ],
            compliance_lst=[ds_compliance, vg_compliance],
            sr830_current_resistor=None,  # only float
            if_combine_gen=True,  # False for coexistence of vary and mapping
            special_name=folder_name,
            sweep_tables=map_lsts,
            measure_nickname="ds-gate-mapping",
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
                    1,
                    plot_types=[["scatter"], ["scatter"], ["contour"]],
                    titles=titles,
                    axes_labels=axes_labels,
                )
            else:
                plotobj.live_plot_init(
                    2,
                    1,
                    1,
                    plot_types=[["scatter"], ["scatter"]],
                    titles=[["n I_{ds}"], ["V_{g} I_{g}"]],
                    axes_labels=[
                        [[r"$n$", r"$I_{ds}$"]],
                        [[r"$V_{g}$", r"$I_{ds}$"]],
                    ],
                )

            plotobj.start_saving(mea_dict["plot_record_path"], 30)

        for i in mea_dict["gen_lst"]:
            self.record_update(mea_dict["file_path"], mea_dict["record_num"], i)
            if not constrained:
                x_data = i[1] if if_inner_loop_is_ds else i[2]
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
                    [0, 1],
                    [0] * 2,
                    [0] * 2,
                    [calculate_from_ds(i[1]), i[2]],
                    [i[3], i[4]],
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
