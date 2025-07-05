#!/usr/bin/env python

"""This module is responsible for managing the measure-related folders and data Note each instrument better be
initialzed right before the measurement, as there may be a long time between loading and measuremnt, leading to
possibilities of parameter changing"""

import copy
import shutil
from itertools import product
from typing import Literal, Generator, Optional, Sequence, Callable
import numpy as np
import pyvisa
import pandas as pd
import re
from functools import partial
import time
from pyomnix.data_process import DataManipulator
from pyomnix.utils import (
    convert_unit,
    gen_seq,
    constant_generator,
    combined_generator_list,
    rename_duplicates,
    time_generator,
)
from pyomnix.omnix_logger import get_logger
from .drivers.probe_rotator import RotatorProbe
from .file_organizer import print_help_if_needed, FileOrganizer
from .equip_wrapper import (
    ITCs,
    ITCMercury,
    WrapperSR830,
    WrapperSR860,
    Wrapper2400,
    Wrapper6430,
    Wrapper2182,
    Wrapper6221,
    Wrapper2450,
    Meter,
    SourceMeter,
    WrapperIPS,
    WrapperB2902Bchannel,
    ITCLakeshore,
    LaserSys,
)
from .equip_wrapper.simulated import SimMeter, SimMag, SimITC, FakeMag, FakeITC
from .constants import SafePath, BoundedCounter

logger = get_logger(__name__)


class MeasureManager(FileOrganizer):
    """This class is a subclass of FileOrganizer and is responsible for managing the measure-related folders and data
    During the measurement, the data will be recorded in self.dfs["curr_measure"], which will be overwritten after
    """

    def __init__(self, proj_name: str) -> None:
        """Note that the FileOrganizer.out_database_init method should be called to assign the correct path to the
        out_database attribute. This method should be called before the MeasureManager object is created."""
        super().__init__(proj_name)  # Call the constructor of the parent class
        self.meter_wrapper_dict = {
            "6430": Wrapper6430,
            "2182": Wrapper2182,
            "2400": Wrapper2400,
            "2450": Wrapper2450,
            "6221": Wrapper6221,
            "sr830": WrapperSR830,
            "sr860": WrapperSR860,
            "b2902ch": WrapperB2902Bchannel,
            "tests": SimMeter
        }
        self.instrs: dict[str, list[Meter] | ITCs | WrapperIPS | RotatorProbe] = {}
        # load params for plotting in measurement
        DataManipulator.load_settings(False, False)
        self.dfs: dict[str, pd.DataFrame] = {}

    @property
    def proj_path(self) -> SafePath:
        """
        return the project path for manual use
        """
        return self._out_database_dir_proj

    def load_meter(
        self,
        meter_no: Literal[
            "sr830", "6221", "2182", "2182a", "2400", "2401", "6430", "2450",
            "b2902ch", "b2902", "b2902b", "tests", "sr860"
        ],
        *address: str,
        channel: int | str = 1,
    ) -> None:
        """
        load the instrument according to the address, store it in self.instrs[meter]
        for tests, use "tests" will load test meters as well as ITC and Magnet

        Args:
            meter_no (str): the name of the instrument
            address (str): the address of the instrument
        """
        # some meters can not be loaded twice, so del old one first
        meter_no = meter_no.lower()
        meter_no.replace("2401", "2400").replace("2182a", "2182").replace("b2902", "b2902ch").replace("b2902b", "b2902ch")

        if meter_no not in self.instrs:
            self.instrs[meter_no] = []
        for addr in address:
            if meter_no == "b2902ch":
                self.instrs[meter_no].append(
                    self.meter_wrapper_dict[meter_no](addr, channel=channel)
                )
            else:
                self.instrs[meter_no].append(self.meter_wrapper_dict[meter_no](addr))
            try:
                self.instrs[meter_no][-1].setup(function="source", reset=True)
            except:
                self.instrs[meter_no][-1].setup(function="sense", reset=True)
        
        if meter_no == "tests":
            self.instrs["itc"] = SimITC()
            self.instrs["ips"] = SimMag()

    def load_rotator(self) -> None:
        """
        load the rotator instrument, store it in self.instrs["rotator"]
        """
        self.instrs["rotator"] = RotatorProbe()
        logger.info("Rotator loaded, please check the status:")
        logger.info("Curr Angle: %s", self.instrs["rotator"].curr_angle())
        logger.info("Curr Velocity: %s", self.instrs["rotator"].spd())

    def load_ITC503(self, gpib_up: str, gpib_down: str) -> None:
        """
        load ITC503 instruments according to the addresses, store them in self.instrs["itc503"] in corresponding order. Also store the ITC503 instruments in self.instrs["itc"] for convenience to call

        Args:
            gpib_up (str): the address of the upper ITC503
            gpib_down (str): the address of the lower ITC503
        """
        self.instrs["itc503"] = ITCs(gpib_up, gpib_down)
        self.instrs["itc"] = self.instrs["itc503"]

    def load_mercury_ips(
        self,
        address: str = "TCPIP0::10.97.24.237::7020::SOCKET",
        if_print: bool = False,
        limit_sphere: float = 11,
    ) -> None:
        """
        load Mercury iPS instrument according to the address, store it in self.instrs["ips"]

        Args:
            address (str): the address of the instrument
            if_print (bool): whether to print the snapshot of the instrument
            limit_sphere (float): the limit of the field
        """
        self.instrs["ips"] = WrapperIPS(
            address, if_print=if_print, limit_sphere=limit_sphere
        )

    def load_laser(self) -> None:
        """
        load the laser instrument, store it in self.instrs["laser"]
        """
        self.instrs["laser"] = LaserSys()

    def load_mercury_itc(
        self,
        address: str = "TCPIP0::10.101.28.24::7020::SOCKET",
        cache_length: int = 60,
        var_crit: float = 5e-4,
    ) -> None:
        """
        load Mercury iPS instrument according to the address, store it in self.instrs["ips"]
        """
        # self.instrs["mercury_itc"] = MercuryITC(address)
        self.instrs["mercury_itc"] = ITCMercury(
            address, cache_length=cache_length, var_crit=var_crit
        )
        self.instrs["itc"] = self.instrs["mercury_itc"]
        # print(self.instrs["mercury_itc"].modules)
    
    def load_fakes(self, no_meters: int = 0) -> None:
        """
        load fake instruments if no real instruments are loaded
        """
        if "ips" not in self.instrs:
            self.instrs["ips"] = FakeMag()
        if "itc" not in self.instrs:
            self.instrs["itc"] = FakeITC()
        if no_meters > 0:
            self.instrs["fakes"] = []
            for i in range(no_meters):
                self.instrs["fakes"].append(SimMeter())

    def load_lakeshore(
        self,
        address: str = "GPIB0::12::INSTR",
        cache_length: int = 60,
        var_crit: float = 5e-4,
    ) -> None:
        """
        load Lakeshore instrument according to the address, store it in self.instrs["lakeshore"]
        """
        self.instrs["lakeshore"] = ITCLakeshore(
            address, cache_length=cache_length, var_crit=var_crit
        )
        self.instrs["itc"] = self.instrs["lakeshore"]

    def source_fixed_apply(self,
                           source_type: Literal["volt", "curr", "V", "I"],
                           value: float | str,
                           *,
                           meter: SourceMeter,
                           compliance: float | str = None,
                           freq: float | str = None,
                           ramp_interval: float | None = None,
                           source_wait: float = 0.1,
                           ) -> Generator[float, None, None]:
        """
        source a fixed value
        """
        #if isinstance(meter, Wrapper6430 | Wrapper2450):
        #    if source_type == "curr":
        #        meter.setup(function="sense", sense_type="volt")
        #    elif source_type == "volt":
        #        meter.setup(function="sense", sense_type="curr")
        if freq is not None:
            freq = convert_unit(freq, "Hz")[0]
        if compliance is not None:
            compliance = convert_unit(compliance, "")[0]
        value = convert_unit(value, "")[0]
        meter.ramp_output(
            source_type,
            value,
            freq=freq,
            compliance=compliance,
            interval=ramp_interval,
            )
        time.sleep(source_wait)
        while True:
            yield value


    def source_sweep_apply(
        self,
        source_type: Literal["volt", "curr", "V", "I"],
        ac_dc: Literal["ac", "dc"],
        meter: str | SourceMeter,
        *,
        max_value: float | str,
        step_value: float | str,
        compliance: float | str,
        freq: float | str = None,
        sweepmode: Optional[
            Literal["0-max-0", "0--max-max-0", "0-max--max-max-0", "0-max", "manual"]
        ] = None,
        resistor: Optional[float] = None,
        sweep_table: Optional[list[float | str, ...]] = None,
        ramp_step: bool = False,
        source_wait: float = 0.1,
        allow_large_jump: bool = False,
    ) -> Generator[float, None, None]:
        """
        source the current using the source meter, initializations will be done automatically

        Args:
            source_type (Literal["volt","curr"]): the type of the source
            ac_dc (Literal["ac","dc"]): the mode of the current
            meter (str | SourceMeter): the meter to be used, use "-0", "-1" to specify the meter if necessary
            max_value (float): the maximum current to be sourced
            step_value (float): the step of the current
            compliance (float): the compliance voltage of the source meter
            freq (float): the frequency of the ac current
            sweepmode (Literal["0-max-0","0--max-max-0","0-max--max-max-0","manual"]): the mode of the dc current sweep, note that the
                "manual" mode is for both ac and dc source, requiring the sweep_table to be provided
            resistor (float): the resistance of the resistor, used only for sr830 source. Once it is provided, the
                source value will be regarded automatically as current
            sweep_table (list[float|str,...]): the table of the sweep values (only if sweepmode is "manual")
            ramp_step (bool): whether to ramp the step value, if true,
              the step value will be ramped to the next value with the interval set by safe_step of each meter
        """
        # load the instrument needed
        source_type = source_type.replace("V", "volt").replace("I", "curr")

        if not allow_large_jump and sweepmode == "manual":
            largest_jump = np.abs(np.diff(sweep_table)).max()
            if source_type == "volt":
                logger.validate(largest_jump < 0.5, f"large jump {largest_jump} in sweep table, if want to force the sweep, set allow_large_jump to True")
            else:
                logger.validate(largest_jump < 0.01, f"large jump {largest_jump} in sweep table, if want to force the sweep, set allow_large_jump to True")

        if meter == "6221" and source_type == "volt":
            raise ValueError("6221 cannot source voltage")
        # for string meter param, could be like "6430"(call the first meter under the type)
        # or "6430-0"(call the first meter under the type), or "6430-1"(call the second meter under the type)
        instr = self.extract_meter_info(meter)

        # convert values to SI and print info
        max_value = convert_unit(max_value, "")[0]
        step_value = convert_unit(step_value, "")[0]
        compliance = convert_unit(compliance, "")[0]
        if freq is not None:
            freq = convert_unit(freq, "Hz")[0]
        logger.info("Source Meter: %s", instr.meter)
        logger.info("Source Type: %s", source_type)
        logger.info("AC/DC: %s", ac_dc)
        logger.info("Max Value: %s %s", max_value, "A" if source_type == "curr" else "V")
        logger.info("Step Value: %s %s", step_value, "A" if source_type == "curr" else "V")
        logger.info("Compliance: %s %s", compliance, "V" if source_type == "curr" else "A")
        logger.info("Freq: %s Hz", freq)
        logger.info("Sweep Mode: %s", sweepmode)
        safe_step: dict | float = instr.safe_step
        if isinstance(safe_step, dict):
            safe_step: float = safe_step[source_type]

        # core functional part
        if ac_dc == "dc":
            if sweepmode == "0-max-0":
                value_gen = self.sweep_values(
                    0, max_value, step_value, mode="start-end-start"
                )
            elif sweepmode == "0--max-max-0":
                value_gen = self.sweep_values(
                    -max_value, max_value, step_value, mode="0-start-end-0"
                )
            elif sweepmode == "0-max--max-max-0":
                value_gen = self.sweep_values(
                    max_value, -max_value, step_value, mode="0-start-end-start-0"
                )
            elif sweepmode == "manual":
                value_gen = (i for i in convert_unit(sweep_table, "")[0])
                instr.ramp_output(
                    source_type,
                    sweep_table[0],
                    interval=safe_step,
                    compliance=compliance,
                )
            else:
                raise ValueError("sweepmode not recognized")
            for value_i in value_gen:
                if ramp_step:
                    instr.ramp_output(
                        source_type,
                        value_i,
                        interval=safe_step,
                        compliance=compliance,
                        no_progress=True,
                    )
                else:
                    instr.uni_output(
                        value_i, compliance=compliance, type_str=source_type
                    )
                time.sleep(source_wait)
                yield value_i
        elif ac_dc == "ac":
            if (
                resistor is not None and source_type == "curr"
            ):  # automatically regard the source value as current and set output mode to volt
                if sweepmode == "manual":
                    volt_gen = (i * resistor for i in convert_unit(sweep_table, "")[0])
                    instr.ramp_output(
                        "volt",
                        sweep_table[0] * resistor,
                        interval=safe_step,
                        compliance=compliance,
                    )
                elif sweepmode == "0-max-0":
                    volt_gen = self.sweep_values(
                        0, max_value * resistor, step_value, mode="start-end-start"
                    )
                elif sweepmode == "0-max":
                    volt_gen = self.sweep_values(
                        0, max_value * resistor, step_value, mode="start-end"
                    )
                else:
                    raise ValueError("sweepmode not recognized")
                for value_i in volt_gen:
                    instr.uni_output(value_i, freq=freq, type_str="volt")
                    time.sleep(source_wait)
                    yield value_i
            else:
                if resistor is not None:
                    logger.warning(
                        "resistor is provided but source type is not current, ignored"
                    )
                if sweepmode == "manual":
                    value_gen = (i for i in convert_unit(sweep_table, "")[0])
                    instr.ramp_output(
                        source_type,
                        sweep_table[0],
                        interval=safe_step,
                        freq=freq,
                        compliance=compliance,
                    )
                elif sweepmode == "0-max-0":
                    value_gen = self.sweep_values(
                        0, max_value, step_value, mode="start-end-start"
                    )
                elif sweepmode == "0-max":
                    value_gen = self.sweep_values(
                        0, max_value, step_value, mode="start-end"
                    )
                else:
                    raise ValueError(f"sweepmode {sweepmode} not recognized")
                for value_i in value_gen:
                    if ramp_step:
                        instr.ramp_output(
                            source_type,
                            value_i,
                            interval=safe_step,
                            freq=freq,
                            compliance=compliance,
                            no_progress=True,
                        )
                    else:
                        instr.uni_output(
                            value_i,
                            freq=freq,
                            compliance=compliance,
                            type_str=source_type,
                        )
                    time.sleep(source_wait)
                    yield value_i

    def ext_sweep_apply(
        self,
        ext_type: Literal["temp", "mag", "B", "T", "angle", "Theta"],
        *,
        min_value: float | str = None,
        max_value: float | str,
        step_value: float | str,
        sweepmode: Literal["0-max-0", "0--max-max-0", "min-max", "manual"] = "0-max-0",
        sweep_table: Optional[tuple[float | str, ...]] = None,
        field_ramp_rate: float = 0.2,
    ) -> Generator[float, None, None]:
        """
        sweep the external field (magnetic/temperature).
        Note that this sweep is the "discrete" sweep, waiting at every point till stabilization

        Args:
            ext_type (Literal["temp","mag"]): the type of the external field
            min_value (float | str): the minimum value of the field
            max_value (float | str): the maximum value of the field
            step_value (float | str): the step of the field
            sweepmode (Literal["0-max-0","0--max-max-0","min-max", "manual"]): the mode of the field sweep
            sweep_table (tuple[float,...]): the table of the sweep values (only if sweepmode is "manual")
            field_ramp_rate (float): the rate of the field ramp (T/min)
        """
        ext_type = (
            ext_type.replace("T", "temp").replace("B", "mag").replace("Theta", "angle")
        )
        if ext_type == "temp":
            instr = self.instrs["itc"]
        elif ext_type == "mag":
            instr = self.instrs["ips"]
        elif ext_type == "angle":
            instr = self.instrs["rotator"]
        else:
            raise ValueError("ext_type not recognized")
        logger.info(f"DISCRETE sweeping mode: {sweepmode}")
        logger.info(f"INSTR: {instr}")
        max_value = convert_unit(max_value, "")[0]
        step_value = convert_unit(step_value, "")[0]
        if min_value is not None:
            min_value = convert_unit(min_value, "")[0]

        if sweepmode == "0-max-0":
            value_gen = self.sweep_values(
                0, max_value, step_value, mode="start-end-start"
            )
        elif sweepmode == "0--max-max-0":
            value_gen = self.sweep_values(
                -max_value, max_value, step_value, mode="0-start-end-0"
            )
        elif sweepmode == "0-max--max-max-0":
            value_gen = self.sweep_values(
                max_value, -max_value, step_value, mode="0-start-end-start-0"
            )
        elif sweepmode == "min-max":
            value_gen = self.sweep_values(
                min_value, max_value, step_value, mode="start-end"
            )
        elif sweepmode == "manual":
            value_gen = (i for i in convert_unit(sweep_table, "")[0])
        else:
            raise ValueError("sweepmode not recognized")

        for value_i in value_gen:
            if ext_type == "temp":
                instr.ramp_to_temperature(value_i, wait=True)
            elif ext_type == "mag":
                instr.ramp_to_field(value_i, rate=field_ramp_rate, wait=True)
            elif ext_type == "angle":
                instr.ramp_angle(value_i)
            yield value_i

    def sense_apply(
        self,
        sense_type: Literal[
            "volt", "curr", "temp", "mag", "V", "I", "T", "B", "H", "angle", "Theta"
        ],
        meter: str | Meter = None,
        *,
        if_during_vary=False,
        vary_criteria: Optional[int | float] = None,
        trigger: Optional[tuple[float, Callable] | float] = None,
        wait_before_vary: int = 7,
    ) -> Generator[float, None, None]:
        """
        sense the current using the source meter, initializations will be done for volt/curr meters
        Currently trigger only supports for temp/mag

        Args:
            sense_type (Literal["volt","curr", "temp","mag"]): the type of the sense
            meter ("str") (applicable only for volt or curr): the meter to be used, use "-0", "-1" to specify the meter if necessary
            if_during_vary (bool): whether the sense is bonded with a varying temp/field, this will limit the generator,
                and the sense will be stopped when the temp/field is stable (not available for meters and rotator)
            vary_criteria (int | float): the criteria (cache variance | no of steps) to judge if the temperature/angle is stable
            trigger (tuple[float, Callable] | float): only applicable when if_during_vary is True, determine the stopping point for varying process. If it's a float, then no trigger function will be applied. If it's a tuple, then the Callable function will be applied when the sense value is stable AND reach the trigger value, the function will be called. Before the trigger, the sense will not be stopped.
        Returns:
            float | tuple[float]: the sensed value (tuple for sr830 ac sense)
        """
        LEAST_TIME_AFTER_TRIGGER = 5 * wait_before_vary if sense_type == "temp" else wait_before_vary # s, used to avoid misjudging right after the trigger
        timer_from_trigger = BoundedCounter(0, min_val=0, max_val=LEAST_TIME_AFTER_TRIGGER) # s, used to avoid misjudging right after the trigger
        timer_before_vary = BoundedCounter(wait_before_vary, min_val=0, max_val=wait_before_vary) # s, used to avoid misjudging before the vary
        if not if_during_vary:
            if trigger is not None:
                logger.warning("trigger is not applicable when if_during_vary is False")
                trigger = None
        else:
            logger.validate(
                trigger is not None,
                "trigger must be provided when if_during_vary is True",
            )
            if isinstance(trigger, tuple):
                trigger_val, trigger_func = trigger
                if_trigger = False
            elif isinstance(trigger, float | int):
                trigger_val = trigger
                if_trigger = True
            else:
                raise ValueError("trigger must be a tuple or a float")

        sense_type = (
            sense_type.replace("V", "volt")
            .replace("I", "curr")
            .replace("T", "temp")
            .replace("B", "mag")
            .replace("H", "mag")
            .replace("Theta", "angle")
        )
        logger.info("Sense Type: %s", sense_type)

        if vary_criteria is not None:
            if vary_criteria < 1:
                logger.info(
                    "variance criteria is set to %s, only suitable for ITC",
                    vary_criteria,
                )
            else:
                logger.warning(
                    "step criteria is deprecated, please use variance criteria instead"
                )
                if sense_type != "angle":
                    vary_criteria = None

        if sense_type in ["volt", "curr"] and meter is not None:
            instr = self.extract_meter_info(meter)
            logger.info("Sense Meter/Instr: %s", instr.meter)
            while True:
                yield instr.sense_delay(type_str=sense_type)

        elif sense_type == "temp":
            instr = self.instrs["itc"]
            logger.info("Sense Meter/Instr: %s", instr)
            if not if_during_vary:
                while True:
                    yield instr.temperature
            else:
                if vary_criteria is not None:
                    instr.set_cache(var_crit=vary_criteria)
                    for _ in range(instr.cache.cache_length):
                        yield instr.temperature
                while (
                    instr.status == "VARYING"
                    or not if_trigger
                    or abs(instr.temperature - trigger_val) > 0.03
                    or timer_from_trigger.count_down()
                    or timer_before_vary.count_down()
                ):
                    yield instr.temperature
                    if (
                        not if_trigger
                        and (abs(trigger_val - instr.temperature) < 0.03)
                        and (instr.status == "HOLD")
                    ):
                        trigger_func()
                        if_trigger = True
                        timer_from_trigger.reset_to_max()
        elif sense_type == "mag":
            instr = self.instrs["ips"]
            logger.info("Sense Meter/Instr: %s", instr)
            if not if_during_vary:
                while True:
                    yield instr.field
            else:
                while (
                    instr.status == "TO SET"
                    or not if_trigger
                    #or abs(instr.field - trigger_val) > 0.01
                    or timer_from_trigger.count_down()
                    or timer_before_vary.count_down()
                ):
                    # only z field is considered
                    yield instr.field
                    if (
                        not if_trigger
                        and (abs(trigger_val - instr.field) < 0.01)
                        and (instr.status == "HOLD")
                    ):
                        trigger_func()
                        if_trigger = True
                        timer_from_trigger.reset_to_max()
        elif sense_type == "angle":
            instr = self.instrs["rotator"]
            logger.info("Sense Meter/Instr: %s", instr)
            if not if_during_vary:
                while True:
                    yield instr.curr_angle()
            else:
                ##TODO: add angle status and trigger
                timer_i = 0
                while timer_i < vary_criteria:
                    # only z field is considered
                    if abs(instr.curr_angle() - instr.angle_set) < 0.03:
                        timer_i += 1
                    else:
                        timer_i = 0
                    yield instr.curr_angle()

    def record_init(
        self,
        measure_mods: tuple[str],
        *var_tuple: float | str,
        manual_columns: Optional[list[str]] = None,
        return_df: bool = False,
        special_folder: str = "",
        measure_nickname: str = "",
        with_timer: bool = True,
        appendix_str: str = "",
    ) -> tuple[SafePath, int, SafePath] | tuple[SafePath, int, pd.DataFrame, SafePath]:
        """
        initialize the record of the measurement and the csv file;
        note the file will be overwritten with an empty dataframe

        Args:
            measure_mods (str): the full name of the measurement (put main source as the first source module term)
            var_tuple (tuple): the variables of the measurement, use "-h" to see the available options
            manual_columns (list[str]): manually appoint the columns (default to None, automatically generate columns)
            return_df (bool): if the final record dataframe will be returned (default not, and saved as a member)
            special_folder (str): the special folder to store the record file (last subfolder, parents[0])
            with_timer (bool): whether to contain time generator
        Returns:
            Path: the file path
            int: the number of columns of the record
        """
        # main_mods, f_str = self.name_fstr_gen(*measure_mods)
        file_path = self.get_filepath(
            measure_mods,
            *var_tuple,
            tmpfolder=special_folder,
            parent_folder=measure_nickname,
            appendix_str=appendix_str,
        )
        tmp_plot_path = self.get_filepath(
            measure_mods,
            *var_tuple,
            parent_folder=measure_nickname,
            tmpfolder=special_folder,
            plot=True,
            suffix=".png",
            appendix_str=appendix_str,
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.add_measurement(*measure_mods)
        logger.info(f"Filename is: {file_path.name}")

        if file_path.exists():
            logger.warning(f"{file_path} already exists, will be overwritten")
            bak_path = file_path.parent / f"{file_path.name}.bak"
            shutil.copy2(file_path, bak_path)
            logger.warning(f"{file_path} has been backed up to {bak_path}")


        mainname_str, _, mod_detail_lst = FileOrganizer.name_fstr_gen(
            *measure_mods, require_detail=True
        )
        # combine the namestr
        pure_name_lst = list(mainname_str.replace("-", "").replace("_", ""))
        if len(pure_name_lst) != len(mod_detail_lst):
            raise ValueError(
                "length of modules doesn't correspond to detail list, check name_fstr_gen method for that"
            )

        if manual_columns is not None:
            columns_lst = manual_columns
        else:
            columns_lst = ["time"] if with_timer else []
            for name, detail in zip(list(pure_name_lst), mod_detail_lst):
                if detail["source_sense"] == "source":
                    columns_lst.append(f"{name}_source")
                elif (
                    detail["source_sense"] == "sense" and detail["ac_dc"] == "ac"
                ):  # note the "sense" is assumed here
                    columns_lst += ["X", "Y", "R", "Theta"]
                else:
                    columns_lst.append(name)

            columns_lst = rename_duplicates(columns_lst)

        self.dfs["curr_measure"] = pd.DataFrame(columns=columns_lst)
        self.dfs["curr_measure"].to_csv(
            file_path, sep=",", index=False, float_format="%.12f"
        )
        if return_df:
            return file_path, len(columns_lst), self.dfs["curr_measure"], tmp_plot_path
        return file_path, len(columns_lst), tmp_plot_path

    def record_update(
        self,
        file_path: SafePath,
        record_num: int,
        record_tuple: tuple[float],
        target_df: Optional[pd.DataFrame] = None,
        force_write: bool = False,
        nocache: bool = True,
    ) -> None:
        """
        update the record of the measurement and also control the size of dataframe
        when the length of current_measure dataframe is larger than 7,

        Args:
            file_path (SafePath): the file path
            record_num (int): the number of columns of the record
            record_tuple (tuple): tuple of the records, with no time column, so length is 1 shorter
            target_df (pd.DataFrame): dataframe to be updated (default using the self.dfs['current_measure'])
            force_write (bool): whether to force write the record
            nocache (bool): whether to keep store all data in memory, if true,
                the dataframe will be written to the file INCREMENTALLY and reset to EMPTY
                (necessary for plotting, only turn on when dataset is extremely large
                    and plotting is not necessary)
        """
        # use reference to ensure synchronization of changes
        if target_df is None:
            curr_df = self.dfs["curr_measure"]
        else:
            curr_df = target_df

        logger.validate(len(record_tuple) == record_num, "The number of columns does not match")
        curr_df.loc[len(curr_df)] = list(record_tuple)
        length = len(curr_df)
        if nocache:
            if length >= 7 or force_write:
                curr_df.to_csv(
                    file_path,
                    sep=",",
                    mode="a",
                    header=False,
                    index=False,
                    float_format="%.12f",
                )
                curr_df.drop(curr_df.index, inplace=True)
        else:
            if (length % 7 == 0) or force_write:
                curr_df.to_csv(file_path, sep=",", index=False, float_format="%.12f")
                # curr_df = pd.DataFrame(columns=curr_df.columns)

    @print_help_if_needed
    def get_measure_dict(
        self,
        measure_mods: tuple[str],
        *var_tuple: float | str,
        wrapper_lst: list[Meter | SourceMeter] = None,
        compliance_lst: list[float | str],
        sr830_current_resistor: float = None,
        if_combine_gen: bool = True,
        sweep_tables: list[list[float | str, ...]]
        | tuple[tuple[float | str, ...]] = None,
        special_name: str = "",
        measure_nickname: str = "",
        with_timer: bool = True,
        no_start_vary: bool = False,
        ramp_intervals: list[float] | tuple[float] = None,
        vary_criteria: Optional[int | float] = None,
        field_ramp_rate: float = 0.2,
        special_mea: Literal["normal", "delta"] = "normal",
        vary_loop: bool = False,
        wait_before_vary: int = 7,
        source_wait: float = 0.05,
        manual_record_columns: Optional[list[str]] = None,
        appendix_str: str = "",
        allow_large_jump: bool = False,
    ) -> dict:
        """
        do the preset of measurements and return the generators, filepath and related info
        1. meter setup should be done before calling this method, they will be bound to generators
        2. the generators are listed in parallel, if there are more than one sweep,
            do manual Cartesian product using itertools.product(gen1,gen2) -> (gen1.1,gen2.all), (gen1.2,gen2.all)...
        3. about the varying of T/B, they will be ramped to the start value first, and then the start_vary functions
            will be returned, call the function to start the varying; and the generator for varying is a sense_apply
        4. no need to appoint sense for ext modules, they will be automatically sensed as long as the module is included

        sweep mode: for I,V-dc: "0-max-0", "0--max-max-0", "manual"
        sweep mode: for I,V-ac: "0-max-0", "0-max", "manual"
        sweep mode: for T,B: "0-max-0", "0--max-max-0", "min-max", "manual"

        Args:
            measure_mods (tuple[str]): the modules of measurement
            var_tuple (tuple): the variables of the measurement, use "-h" to see the variables' list
            wrapper_lst (list[Meter]): the list of the wrappers to be used (only for source and sense)
            compliance_lst (list[float]): the list of the compliance to be used (sources)
            sr830_current_resistor (float): the resistance of the resistor, used only for sr830 curr source
            if_combine_gen (bool): whether to combine the generators as a whole list generator,
                                if False, return the list of generators for further operations
                                (useful for combination of VARY and SWEEP)
            sweep_tables (list[list[float | str, ...]]): the list of the sweep tables for manual sweep,
                                the table will be fetched and used according to the order from left to right(0->1->2...)
            special_name (str): the special name used for subfolder to avoid mixing under the same measurement name
            with_timer (bool): whether to contain time generator
            no_start_vary (bool): vary without starting from a fixed start
            ramp_intervals (list[float]): the intervals for ramping the source, used with care, note the correspondence
            vary_criteria (deprecated): the criteria (no of steps) to judge if the field/temperature is stable
            field_ramp_rate (float): the rate of the field ramp (T/min)
            special_mea (Literal["normal", "delta"]): whether to do the special measurement, "delta" means the delta current-reversal measurement
            vary_loop (bool): whether to loop the varying, if True, the varying will be looped
            wait_before_vary (int): the wait time before varying
            source_wait (float): the wait time after source changes
            manual_record_columns (list[str]): the columns to be manually recorded, used for special measurement

        Returns:
            dict: a dictionary containing the list of generators, dataframe csv filepath and record number
                keys: "gen_lst"(combined list generator), "swp_idx" (indexes for sweeping generator, not including vary),
                "file_path"(csv file), "record_num"(num of record data columns, without time),
                "tmp_vary", "mag_vary", "angle_vary"
                (the function used to begin the varying of T/B/Theta,
                    e.g. start magnetic field varying by calling mag_vary(),
                    add reverse=True to reverse the varying direction, used to do circular varying)
        """
        if special_mea == "delta":
            logger.info(
                "use instance.instrs['6221'][0].delta_setup(**kwargs) to set customized parameters if needed AFTER this method"
            )
            logger.info(
                "delta measurement should use a fixed current, make sure the 6221 source is fixed"
            )

        if sweep_tables is not None:
            if isinstance(sweep_tables, list):
                if isinstance(sweep_tables[0], list):
                    pass
                elif isinstance(sweep_tables[0], np.ndarray):
                    sweep_tables = [i.tolist() for i in sweep_tables]
                elif isinstance(sweep_tables[0], tuple):
                    sweep_tables = [list(i) for i in sweep_tables]
                elif sweep_tables[0] is None:
                    sweep_tables = None
                else:
                    raise TypeError("unsupported sweep_tables type")
            elif isinstance(sweep_tables, tuple):
                return self.get_measure_dict(
                    measure_mods,
                    *var_tuple,
                    wrapper_lst=wrapper_lst,
                    compliance_lst=compliance_lst,
                    sr830_current_resistor=sr830_current_resistor,
                    if_combine_gen=if_combine_gen,
                    sweep_tables=list(sweep_tables),
                    special_name=special_name,
                    with_timer=with_timer,
                    no_start_vary=no_start_vary,
                    ramp_intervals=ramp_intervals,
                    vary_criteria=vary_criteria,
                    field_ramp_rate=field_ramp_rate,
                    special_mea=special_mea,
                    vary_loop=vary_loop,
                    measure_nickname=measure_nickname,
                    wait_before_vary=wait_before_vary,
                    source_wait=source_wait,
                    appendix_str=appendix_str,
                    allow_large_jump=allow_large_jump,
                )
            elif isinstance(sweep_tables, np.ndarray):
                return self.get_measure_dict(
                    measure_mods,
                    *var_tuple,
                    wrapper_lst=wrapper_lst,
                    compliance_lst=compliance_lst,
                    sr830_current_resistor=sr830_current_resistor,
                    if_combine_gen=if_combine_gen,
                    sweep_tables=sweep_tables.tolist(),
                    special_name=special_name,
                    with_timer=with_timer,
                    no_start_vary=no_start_vary,
                    ramp_intervals=ramp_intervals,
                    vary_criteria=vary_criteria,
                    field_ramp_rate=field_ramp_rate,
                    special_mea=special_mea,
                    vary_loop=vary_loop,
                    measure_nickname=measure_nickname,
                    wait_before_vary=wait_before_vary,
                    source_wait=source_wait,
                    appendix_str=appendix_str,
                    allow_large_jump=allow_large_jump,
                )
            else:
                raise TypeError("unsupported sweep_tables type")

        src_lst, sense_lst, oth_lst = self.extract_info_mods(measure_mods, *var_tuple)
        assert len(src_lst) + len(sense_lst) == len(wrapper_lst), (
            "The number of modules and meters should be the same"
        )
        assert len(src_lst) == len(compliance_lst), (
            "The number of sources and compliance should be the same"
        )

        # init record dataframe
        file_path, record_num, record_plot_path = self.record_init(
            measure_mods,
            *var_tuple,
            special_folder=special_name,
            measure_nickname=measure_nickname,
            manual_columns=manual_record_columns,
            appendix_str=appendix_str,
        )
        rec_lst = [time_generator()] if with_timer else []

        # =============assemble the record generators into one list==============
        # note multiple sweeps result in multidimensional mapping
        sweep_idx = []
        vary_mod = []  # T, B, angle
        # source part
        mod_i: Literal["I", "V"]
        for idx, src_mod in enumerate(src_lst):

            if src_mod["I"]["sweep_fix"] is not None:
                mod_i = "I"
            elif src_mod["V"]["sweep_fix"] is not None:
                mod_i = "V"
            else:
                raise ValueError(f"No source is specified for source {idx}")

            if isinstance(wrapper_lst[idx], Wrapper6221):
                wrapper_lst[idx].setup(
                    function="source", mea_mode=special_mea
                )  # here assume only one 6221
            else:
                wrapper_lst[idx].setup(function="source", source_type=mod_i)

            if src_mod[mod_i]["sweep_fix"] == "fixed":
                if ramp_intervals is not None:
                    interval = ramp_intervals.pop(0)
                else:
                    interval = None
                rec_lst.append(self.source_fixed_apply(
                    mod_i,
                    src_mod[mod_i]["fix"],
                    meter=wrapper_lst[idx],
                    compliance=compliance_lst[idx],
                    freq=src_mod[mod_i]["freq"],
                    source_wait=source_wait,
                    ramp_interval=interval,
                ))
            elif src_mod[mod_i]["sweep_fix"] == "sweep":
                if src_mod[mod_i]["mode"] == "manual":
                    sweep_table = sweep_tables.pop(0)
                else:
                    sweep_table = None
                rec_lst.append(
                    self.source_sweep_apply(
                        mod_i,
                        src_mod[mod_i]["ac_dc"],
                        wrapper_lst[idx],
                        max_value=src_mod[mod_i]["max"],
                        step_value=src_mod[mod_i]["step"],
                        compliance=compliance_lst[idx],
                        freq=src_mod[mod_i]["freq"],
                        sweepmode=src_mod[mod_i]["mode"],
                        resistor=sr830_current_resistor,
                        sweep_table=sweep_table,
                        source_wait=source_wait,
                        allow_large_jump=allow_large_jump,
                    )
                )
                sweep_idx.append(idx)
        # sense part
        for idx, sense_mod in enumerate(sense_lst):
            if isinstance(wrapper_lst[idx + len(src_lst)], WrapperSR830) and sense_mod["type"] == "curr":
                wrapper_lst[idx + len(src_lst)].setup(
                    function="sense",
                    input_config="I (1 MOhm)",
                    input_grounding="Ground",
                )
            else:
                sense_type = sense_mod["type"].replace("I", "curr").replace("V", "volt")
                wrapper_lst[idx + len(src_lst)].setup(function="sense", sense_type=sense_type)

            rec_lst.append(
                self.sense_apply(sense_mod["type"], wrapper_lst[idx + len(src_lst)])
            )
        # others part
        for idx, oth_mod in enumerate(oth_lst):
            if oth_mod["sweep_fix"] == "fixed":
                if oth_mod["name"] == "T":
                    self.instrs["itc"].ramp_to_temperature(oth_mod["fix"], wait=True)
                elif oth_mod["name"] == "B":
                    self.instrs["ips"].ramp_to_field(oth_mod["fix"], wait=True)
                elif oth_mod["name"] == "Theta":
                    self.instrs["rotator"].ramp_angle(oth_mod["fix"], wait=True)
                rec_lst.append(self.sense_apply(oth_mod["name"]))
            elif oth_mod["sweep_fix"] == "vary":
                if oth_mod["name"] == "T":
                    vary_mod.append("T")
                    vary_bound_T = (oth_mod["start"], oth_mod["stop"])
                    if not no_start_vary:
                        self.instrs["itc"].ramp_to_temperature(
                            oth_mod["start"], wait=True
                        )

                        def temp_vary(reverse: bool = False, oth_mod=oth_mod):
                            target = oth_mod["start"] if reverse else oth_mod["stop"]
                            ini = oth_mod["stop"] if reverse else oth_mod["start"]
                            while abs(self.instrs["itc"].temperature - ini) > 0.1:
                                self.instrs["itc"].ramp_to_temperature(ini, wait=True)
                            self.instrs["itc"].ramp_to_temperature(target, wait=False)

                    # define a function instead of directly calling the ramp_to_temperature method
                    # to avoid possible interruption or delay
                    else:

                        def temp_vary(reverse: bool = False, oth_mod=oth_mod):
                            target = oth_mod["start"] if reverse else oth_mod["stop"]
                            self.instrs["itc"].ramp_to_temperature(target, wait=False)

                    if vary_loop:
                        trigger_tuple = (
                            oth_mod["stop"],
                            partial(temp_vary, reverse=True),
                        )
                    else:
                        trigger_tuple = oth_mod["stop"]

                elif oth_mod["name"] == "B":
                    vary_mod.append("B")
                    self.instrs["ips"].ramp_to_field(oth_mod["start"], wait=True)
                    vary_bound_B = (oth_mod["start"], oth_mod["stop"])

                    def mag_vary(reverse: bool = False, oth_mod=oth_mod):
                        target = oth_mod["start"] if reverse else oth_mod["stop"]
                        ini = oth_mod["stop"] if reverse else oth_mod["start"]
                        while abs(float(self.instrs["ips"].field) - ini) > 0.01:
                            self.instrs["ips"].ramp_to_field(ini, wait=True)
                        self.instrs["ips"].ramp_to_field(
                            target, rate=field_ramp_rate, wait=False
                        )

                    if vary_loop:
                        trigger_tuple = (
                            oth_mod["stop"],
                            partial(mag_vary, reverse=True),
                        )
                    else:
                        trigger_tuple = oth_mod["stop"]

                elif oth_mod["name"] == "Theta":
                    vary_mod.append("Theta")
                    self.instrs["rotator"].ramp_angle(oth_mod["start"], wait=True)
                    vary_bound_Theta = (oth_mod["start"], oth_mod["stop"])

                    def angle_vary(reverse: bool = False, oth_mod=oth_mod):
                        target = oth_mod["start"] if reverse else oth_mod["stop"]
                        ini = oth_mod["stop"] if reverse else oth_mod["start"]
                        while abs(self.instrs["rotator"].curr_angle() - ini) > 0.3:
                            self.instrs["rotator"].ramp_angle(ini, wait=True)
                        self.instrs["rotator"].ramp_angle(target, wait=False)

                    if vary_loop:
                        trigger_tuple = (
                            oth_mod["stop"],
                            partial(angle_vary, reverse=True),
                        )
                    else:
                        trigger_tuple = oth_mod["stop"]

                else:
                    raise ValueError("Vary module not recognized")

                rec_lst.append(
                    self.sense_apply(
                        oth_mod["name"],
                        if_during_vary=True,
                        vary_criteria=vary_criteria,
                        trigger=trigger_tuple,
                        wait_before_vary=wait_before_vary,
                    )
                )
            elif oth_mod["sweep_fix"] == "sweep":
                if oth_mod["mode"] == "manual":
                    sweep_table = sweep_tables.pop(0)
                else:
                    sweep_table = None
                rec_lst.append(
                    self.ext_sweep_apply(
                        oth_mod["name"],
                        min_value=oth_mod["min"],
                        max_value=oth_mod["max"],
                        step_value=oth_mod["step"],
                        sweepmode=oth_mod["mode"],
                        sweep_table=sweep_table,
                    )
                )
                sweep_idx.append(idx + len(src_lst) + len(sense_lst))
        if if_combine_gen:
            total_gen = combined_generator_list(rec_lst)
        else:
            total_gen = rec_lst

        if with_timer:
            sweep_idx = [i + 1 for i in sweep_idx]  # add 1 for the time column
        return {
            "gen_lst": total_gen,
            "swp_idx": sweep_idx,
            "file_path": file_path,
            "plot_record_path": record_plot_path,
            "record_num": record_num,
            "vary_mod": vary_mod,
            "tmp_vary": None
            if "T" not in vary_mod
            else (
                temp_vary,
                lambda: self.instrs["itc"].temperature,
                lambda: self.instrs["itc"].temperature_set,
                vary_bound_T,
            ),
            "mag_vary": None
            if "B" not in vary_mod
            else (
                mag_vary,
                lambda: self.instrs["ips"].field,
                lambda: self.instrs["ips"].field_set,
                vary_bound_B,
            ),
            "angle_vary": None
            if "Theta" not in vary_mod
            else (
                angle_vary,
                self.instrs["rotator"].curr_angle,
                lambda: self.instrs["rotator"].angle_set,
                vary_bound_Theta,
            ),
        }

    def watch_sense(
        self,
        sense_mods: tuple[str],
        time_len: Optional[int] = None,
        time_step: int = 1,
        filename: str | SafePath = "tmp",
        wrapper_lst: list[Meter] = None,
    ) -> tuple[SafePath, int, Generator[tuple[float], None, None], list[str]]:
        """
        watch the sense values with time and record them into the csv file
        this function is basically a special case of get_measure_dict method
        with only sense modules and no other modules

        Args:
            sense_mods (tuple[str]): targeting modules (only main str needed,
                like T / temp, B / mag, V / volt, I / curr)
            time_len (int (s) | None): the time length of the measurement (None = Inf)
            time_step (int(s)): the time step of the measurement, default to 1s
            filename (str): the filename of the csv file (parent path will be project / watch)
            wrapper_lst (list[Meter]): the list of the wrappers to be used
        """
        # only need the main name of the sense module
        sense_mods = [
            sense_mods[i].split("_")[0].split("-")[0] for i in range(len(sense_mods))
        ]
        file_path = self.proj_path / "watch" / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        rec_lst = [time_generator()]
        cols = ["time"]
        for sense_mod in sense_mods:
            sense_mod = (
                sense_mod.replace("T", "temp")
                .replace("B", "mag")
                .replace("H", "mag")
                .replace("Theta", "angle")
            )
            if sense_mod not in ["mag", "temp", "angle"]:
                rec_lst.append(
                    self.sense_apply(sense_mod, tmp_wrapper := wrapper_lst.pop(0))
                )
            else:
                rec_lst.append(self.sense_apply(sense_mod))

            if sense_mod == "V" and isinstance(tmp_wrapper, WrapperSR830):
                cols += ["X", "Y", "R", "Theta"]
            else:
                cols.append(sense_mod)

        cols = rename_duplicates(cols)
        total_gen = combined_generator_list(rec_lst)

        self.dfs["curr_measure"] = pd.DataFrame(columns=cols)
        self.dfs["curr_measure"].to_csv(
            file_path, sep=",", index=False, float_format="%.12f"
        )

        return file_path, len(cols), total_gen, cols

    def extract_meter_info(self, meter: str | Meter) -> Meter | SourceMeter:
        """
        convert the meter name to the meter object and print the name of object
        (directly return it if it is already a meter object)
        """
        if isinstance(meter, str):
            if len(meter.split("-")) == 1:
                instr = self.instrs[meter][0]
            elif len(meter_tuple := meter.split("-")) == 2:
                instr = self.instrs[meter_tuple[0]][int(meter_tuple[1])]
            else:
                raise ValueError("meter name is not in the correct format")
        elif isinstance(meter, Meter):
            instr = meter
        else:
            raise ValueError("meter name not recognized")
        return instr

    @staticmethod
    def create_mapping(
        *lists: Sequence[float | str], idxs: Sequence[int] = None
    ) -> tuple[tuple[float | str, ...]]:
        """
        create the mapping of the lists, return the tuple of the mapping
        e.g.: create_mapping([1,2],[4,6], idx=(0,1)) runs mapping effectively ((1,4),(1,6),(2,4),(2,6)) (this is the final effect, not the returned value)

        Args:
            lists (Sequence): the lists to be mapped
            idxs (Sequence): (from 0) the indexes of the lists (the first index corresponds to the first list)
        """
        if idxs is None:
            rearrange_lsts = lists
        else:
            rearrange_lsts = [[]] * len(lists)
            for n, idx in enumerate(idxs):
                rearrange_lsts[idx] = lists[n]
        mat = product(*rearrange_lsts)
        mat_cols = tuple(zip(*mat))
        if idxs is not None:
            restore_lsts = tuple([mat_cols[i] for i in idxs])
        else:
            restore_lsts = mat_cols
        return restore_lsts

    @staticmethod
    def get_visa_resources() -> tuple[str, ...]:
        """
        return a list of visa resources
        """
        return pyvisa.ResourceManager().list_resources()

    @staticmethod
    def write_header(file_path: SafePath, header: str) -> None:
        """
        write the header to the file

        Args:
            file_path (str): the file path
            header (str): the header to write
        """
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(header)

    @staticmethod
    def sweep_values(
        start_value: float,
        end_value: float,
        step: float,
        mode: Literal[
            "start-end", "start-end-start", "0-start-end-0", "0-start-end-start-0"
        ],
    ) -> Generator[float, None, None]:
        """
        generate sweeping sequence according to the mode
        NOTE: the values at ends will be repeated
        """
        if mode == "start-end":
            yield from gen_seq(start_value, end_value, step)
        elif mode == "start-end-start":
            yield from gen_seq(start_value, end_value, step)
            yield from gen_seq(end_value, start_value, -step)
        elif mode == "0-start-end-0":
            yield from gen_seq(0, start_value, step)
            yield from gen_seq(start_value, end_value, step)
            yield from gen_seq(end_value, 0, -step)

        elif mode == "0-start-end-start-0":
            yield from gen_seq(0, start_value, step)
            yield from gen_seq(start_value, end_value, step)
            yield from gen_seq(end_value, start_value, -step)
            yield from gen_seq(start_value, 0, -step)

    @staticmethod
    def extract_info_mods(
        measure_mods: tuple[str], *var_tuple: float | str
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Extract the information from the measure_mods and var_tuple
        """
        main_mods, f_str, mod_detail_lst = FileOrganizer.name_fstr_gen(
            *measure_mods, require_detail=True
        )

        # ================= 1. Load parameters from name str and var_tuple =================
        # this step can be manually done by the user
        def find_positions(lst: list[str], search_term: str) -> int:
            """
            Find positions of elements in the list that contain the search term as a substring.

            Args:
                lst (list[str]): The list to search.
                search_term (str): The term to search for.

            Returns:
                list[int]: The list of positions where the search term is found.
            """
            return [i for i, element in enumerate(lst) if search_term in element][0]

        src_no, sense_no, oth_no = list(map(len, main_mods.split("-")))
        mods_lst = list(main_mods.replace("-", "").replace("_", ""))
        source_dict = {
            "I": {
                "sweep_fix": None,
                "ac_dc": None,
                "fix": None,
                "max": None,
                "step": None,
                "mode": None,
                "freq": None,
            },
            "V": {
                "sweep_fix": None,
                "ac_dc": None,
                "fix": None,
                "max": None,
                "step": None,
                "mode": None,
                "freq": None,
            },
        }
        sense_dict = {"type": None, "ac_dc": None}
        other_dict = {
            "name": None,
            "sweep_fix": None,
            "fix": None,
            "start": None,
            "stop": None,
            "step": None,
            "mode": None,
        }
        src_lst = [copy.deepcopy(source_dict) for _ in range(src_no)]
        sense_lst = [copy.deepcopy(sense_dict) for _ in range(sense_no)]
        other_lst = [copy.deepcopy(other_dict) for _ in range(oth_no)]
        # the index to retrieve the variables from var_tuple
        index_vars = 0
        for idx, (mod, detail) in enumerate(zip(mods_lst, mod_detail_lst)):
            if idx < src_no:
                vars_lst = re.findall(
                    r"{(\w+)}",
                    MeasureManager.measure_types_json[mod]["source"][
                        detail["sweep_fix"]
                    ][detail["ac_dc"]],
                )
                length = len(vars_lst)
                src_lst[idx][mod]["ac_dc"] = detail["ac_dc"]
                src_lst[idx][mod]["sweep_fix"] = detail["sweep_fix"]
                if detail["ac_dc"] == "ac":
                    src_lst[idx][mod]["freq"] = var_tuple[
                        index_vars + find_positions(vars_lst, "freq")
                    ]
                if detail["sweep_fix"] == "sweep":
                    src_lst[idx][mod]["max"] = var_tuple[
                        index_vars + find_positions(vars_lst, "max")
                    ]
                    src_lst[idx][mod]["step"] = var_tuple[
                        index_vars + find_positions(vars_lst, "step")
                    ]
                    src_lst[idx][mod]["mode"] = var_tuple[
                        index_vars + find_positions(vars_lst, "mode")
                    ]
                elif detail["sweep_fix"] == "fixed":
                    src_lst[idx][mod]["fix"] = var_tuple[
                        index_vars + find_positions(vars_lst, "fix")
                    ]
            elif idx < src_no + sense_no:
                vars_lst = re.findall(
                    r"{(\w+)}", MeasureManager.measure_types_json[mod]["sense"][detail["ac_dc"]]
                )
                length = len(vars_lst)
                sense_lst[idx - src_no]["type"] = mod
                sense_lst[idx - src_no]["ac_dc"] = detail["ac_dc"]
            else:
                vars_lst = re.findall(
                    r"{(\w+)}",
                    MeasureManager.measure_types_json[mod][detail["sweep_fix"]],
                )
                length = len(vars_lst)
                other_lst[idx - src_no - sense_no]["name"] = mod
                other_lst[idx - src_no - sense_no]["sweep_fix"] = detail["sweep_fix"]
                if detail["sweep_fix"] == "sweep":
                    other_lst[idx - src_no - sense_no]["start"] = var_tuple[
                        index_vars + find_positions(vars_lst, "start")
                    ]
                    other_lst[idx - src_no - sense_no]["stop"] = var_tuple[
                        index_vars + find_positions(vars_lst, "stop")
                    ]
                    other_lst[idx - src_no - sense_no]["step"] = var_tuple[
                        index_vars + find_positions(vars_lst, "step")
                    ]
                    other_lst[idx - src_no - sense_no]["mode"] = var_tuple[
                        index_vars + find_positions(vars_lst, "mode")
                    ]
                elif detail["sweep_fix"] == "fixed":
                    other_lst[idx - src_no - sense_no]["fix"] = var_tuple[
                        index_vars + find_positions(vars_lst, "fix")
                    ]
                elif detail["sweep_fix"] == "vary":
                    other_lst[idx - src_no - sense_no]["start"] = var_tuple[
                        index_vars + find_positions(vars_lst, "start")
                    ]
                    other_lst[idx - src_no - sense_no]["stop"] = var_tuple[
                        index_vars + find_positions(vars_lst, "stop")
                    ]

            index_vars += length

        return src_lst, sense_lst, other_lst
