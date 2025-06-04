#!/usr/bin/env python

"""
This module contains the wrapper classes for used equipments in
measure_manager.py. The purpose of this module is to unify the interface
of different equipments, so that they can be combined freely

! not all equipments are included in this module, only those needed wrapping
! for some equipments already wrapped(probe_rotator), the wrapper is not necessary

each wrapper provides the following methods(some only for source meters):
- setup: initialize the equipment, usually just basic settings not including output
- output_switch: switch the output on or off
- uni_output: set the output to a certain value
    all output methods have two implementations, one is from off to on, including
    setting up parameters like range and compliance, the other is just setting the
    output value when already on
- get_output_status: get the current output value
- sense: set the meter to sense current or voltage
- shutdown: shutdown the equipment
- ramp_output: ramp the output to the target value
* the member "meter" is provided for directly accessing the equipment driver
* the member "info_dict" is provided for storing the information of the equipment

Flow:
    Wrapperxxxx(GPIB)
    setup("ac/dc")
    uni_output(value, (freq), compliance, type_str)
    (change value without disabling the output)
    shutdown()

Actions that can be optimized for quicker operation:
- switch operation
- range and compliance setting
"""

import time
from typing import Literal, Optional

from abc import ABC, abstractmethod

import numpy as np
from pymeasure.instruments.srs import SR830
from pymeasure.instruments.keithley import KeithleyDMM6500
from pymeasure.instruments.keithley import Keithley2182
from qcodes.instrument_drivers.Keithley import Keithley2400, Keithley2450
from qcodes.instrument import find_or_create_instrument
from pyomnix.omnix_logger import get_logger
from pyomnix.utils import convert_unit, print_progress_bar, SWITCH_DICT

from ..drivers.Keithley_6430 import Keithley_6430
from ..drivers.keithley6221 import Keithley6221
from ..drivers.keysight_b2902b import Keysight_B2902B, KeysightB2902BChannel
from ..drivers.SR860 import SR860

logger = get_logger(__name__)


class Meter(ABC):
    """
    The usage should be following the steps:
    1. instantiate
    2. setup method
    3. output_switch method (embedded in output method for the first time)
    4. uni/rms/dc_output method or ramp_output method
    5(if needed). sense method
    LAST. shutdown method
    """

    @abstractmethod
    def __init__(self):
        self.info_dict = {}
        self.meter = None

    @abstractmethod
    def setup(self, function: Literal["sense", "source"], *vargs, **kwargs):
        pass

    def info(self, *, sync=True):
        if sync:
            self.info_sync()
        return self.info_dict

    @abstractmethod
    def info_sync(self):
        self.info_dict.update({})

    def sense_delay(self, type_str: Literal["curr", "volt"], *, delay: float = 0.01):
        time.sleep(delay)
        return self.sense(type_str=type_str)

    @abstractmethod
    def sense(self, type_str: Literal["curr", "volt"]) -> float | list:
        pass

    @property
    @abstractmethod
    def sense_range_volt(self) -> float:
        pass

    @sense_range_volt.setter
    @abstractmethod
    def sense_range_volt(self, fix_range: float):
        pass

    @property
    @abstractmethod
    def sense_range_curr(self) -> float:
        pass

    @sense_range_curr.setter
    @abstractmethod
    def sense_range_curr(self, fix_range: float):
        pass

    def chg_sense_range(
        self, fix_range: float | str, type_str: Literal["curr", "volt", "all"] = "all"
    ) -> float | list:
        """
        change the sense range of the meter
        type_str: "curr" or "volt" or "all"
        fix_range: the range to be set, if None, then auto range is used
        """
        if isinstance(fix_range, str):
            fix_range = convert_unit(fix_range, "")[0]
        if type_str == "all":
            self.sense_range_volt = fix_range
            self.sense_range_curr = fix_range
        elif type_str == "curr":
            self.sense_range_curr = fix_range
        elif type_str == "volt":
            self.sense_range_volt = fix_range

    def __del__(self):
        try:
            self.meter.__del__()
        except AttributeError:
            del self.meter


class SourceMeter(Meter):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self.info_dict.update({"output_type": "curr"})
        self.output_target = 0
        self.safe_step = 1e-6  # step threshold, used for a default ramp

    @abstractmethod
    def output_switch(self, switch: bool | Literal["on", "off", "ON", "OFF"]):
        """the meter must be returned to 0"""
        self.info_dict["output_status"] = (
            switch if isinstance(switch, bool) else switch.lower() in ["on", "ON"]
        )

    @property
    def source_range(self) -> float:
        logger.warning("source_range not implemented")

    @source_range.setter
    def source_range(self, fix_range: float):
        logger.warning("source_range.setter not implemented")

    @property
    def compliance(self) -> float:
        logger.warning("compliance not implemented")

    @compliance.setter
    def compliance(self, fix_compliance: float):
        logger.warning("compliance.setter not implemented")

    @abstractmethod
    def uni_output(
        self,
        value: float | str,
        *,
        freq: Optional[float | str] = None,
        compliance: Optional[float | str] = None,
        fix_range: Optional[float | str] = None,
        alter_range: bool = False,
        type_str: Literal["curr", "volt"],
    ) -> float:
        """
        judge the output type based on if freq is none
        judge if the range and compliance need to be set or modified
        (only modify if needed)
        return the real output value to avoid range issue etc.
        """
        self.info_dict["output_type"] = type_str
        self.output_target = value

        return self.get_output_status()[0]

    @abstractmethod
    def get_output_status(self) -> tuple[float, float, float] | tuple[float, float]:
        """
        return the output value from device and also the target value set by output methods

        Returns:
            tuple[float, float, float]: the actual output value and the target value and current range
            or tuple[float, float]: the actual output value and the target value (no range, e.g. for sr830)
        """
        pass

    @abstractmethod
    def shutdown(self):
        pass

    def ramp_output(
        self,
        type_str: Literal["curr", "volt", "V", "I"],
        value: float | str,
        *,
        freq: Optional[float | str] = None,
        compliance: Optional[float | str] = None,
        interval: Optional[float | str] = None,
        sleep=0.2,
        from_curr=True,
        no_progress=False,
    ) -> None:
        """
        ramp the output to the target value

        Args:
            type_str: "curr" or "volt"
            value: the target value
            freq: the frequency of the output (if ac)
            interval: the step interval between each step
            sleep: the time interval(s) between each step
            value: the target value
            compliance: the compliance value
            from_curr: whether to ramp from the current value(default) or from 0
            no_progress: whether to suppress the progress bar
        """
        type_str: Literal["curr", "volt"] = type_str.replace("V", "volt").replace(
            "I", "curr"
        )
        value = convert_unit(value, "")[0]
        if not from_curr:
            # reset the output to 0 (ensure it in output_switch method)
            self.output_switch("off")
            self.output_switch("on")

        curr_val = self.get_output_status()[0]
        if curr_val == value:
            self.uni_output(
                value,
                freq=freq,
                type_str=type_str,
                compliance=compliance,
                alter_range=True,
            )
            return
        if interval is None:
            if abs(curr_val - value) > 20:
                arr = np.arange(curr_val, value, 0.2 * np.sign(value - curr_val))
            else:
                arr = np.linspace(curr_val, value, 70)
        elif isinstance(interval, (float, str)):
            interval = convert_unit(interval, "")[0]
            interval = abs(interval) * np.sign(value - curr_val)
            arr = np.arange(curr_val, value, interval)
            arr = np.concatenate((arr, [value]))
        else:
            raise ValueError(
                "interval should be a float or str or just left as default"
            )

        for idx, i in enumerate(arr):
            self.uni_output(
                i, freq=freq, type_str=type_str, compliance=compliance, alter_range=True
            )
            if not no_progress:
                print_progress_bar(
                    (idx + 1) / len(arr) * 100, 100, prefix="Ramping Meter:"
                )
            time.sleep(sleep)


class ACSourceMeter(SourceMeter):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def rms_output(
        self,
        value: float | str,
        *,
        freq: float | str,
        compliance: float | str,
        type_str: Literal["curr", "volt"],
    ):
        pass


class DCSourceMeter(SourceMeter):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def dc_output(
        self,
        value: float | str,
        *,
        compliance: float | str,
        type_str: Literal["curr", "volt"],
        fix_range: Optional[float | str] = None,
    ):
        pass


class Wrapper6221(ACSourceMeter, DCSourceMeter):
    """
    Flow:
    Wrapper6221(GPIB)
    setup("ac/dc")
    uni_output(value, (freq), compliance, type_str)
    (change value without disabling the output)
    shutdown()
    """

    def __init__(self, GPIB: str = "GPIB0::12::INSTR"):
        super().__init__()
        self.meter = Keithley6221(GPIB)
        self.output_target = 0
        self.safe_step = 1e-6
        self.info_dict = {
            "GPIB": GPIB,
            "output_type": "curr",
            "ac_dc": "ac",
            "output_status": False,
            "output_value": 0,
        }
        self.info_sync()
        self.mea_mode: Literal["normal", "delta", "pulse-delta", "differential"] = (
            "normal"
        )
        logger.info("note the grounding:")  # TODO: add grounding instruction#

    def info_sync(self):
        self.info_dict.update(
            {
                "source_range": self.meter.source_range,
                "source_range_set": 0,
                "source_compliance_set": 0,
                "output_value": max(
                    self.meter.source_current, self.meter.waveform_amplitude
                ),
                "frequency": self.meter.waveform_frequency,
                "compliance": self.meter.source_compliance,
                "wave_function": self.meter.waveform_function,
                "wave_offset": self.meter.waveform_offset,
                "wave_phasemarker": self.meter.waveform_phasemarker_phase,
                "low_grounded": self.meter.output_low_grounded,
            }
        )

    @property
    def sense_range_volt(self) -> float:
        logger.warning("sense_range_volt not implemented for 6221")

    @sense_range_volt.setter
    def sense_range_volt(self, fix_range: float):
        logger.warning("sense_range_volt.setter not implemented for 6221")

    @property
    def sense_range_curr(self) -> float:
        logger.warning("sense_range_curr not implemented for 6221")

    @sense_range_curr.setter
    def sense_range_curr(self, fix_range: float):
        logger.warning("sense_range_curr.setter not implemented for 6221")

    @property
    def compliance(self) -> float:
        return self.meter.source_compliance

    @compliance.setter
    def compliance(self, fix_compliance: float):
        if self.info_dict["source_compliance_set"] != fix_compliance:
            self.meter.source_compliance = fix_compliance
            self.info_dict["source_compliance_set"] = fix_compliance

    @property
    def source_range(self) -> float:
        self.info_dict.update({"source_range": self.meter.source_range})
        return self.meter.source_range

    @source_range.setter
    def source_range(self, fix_range: float):
        if self.info_dict["source_range_set"] != fix_range:
            self.meter.source_range = fix_range
            self.info_dict["source_range_set"] = fix_range

    def setup(
        self,
        function: Literal["source", "sense"] = "source",
        mode: Literal["ac", "dc"] = "ac",
        *,
        offset: float | None = None,
        source_auto_range: bool | None = None,
        low_grounded: bool | None = None,
        wave_function: Literal[
            "sine",
            "ramp",
            "square",
            "arbitrary1",
            "arbitrary2",
            "arbitrary3",
            "arbitrary4",
        ]
        | None = None,
        mea_mode: Literal["normal", "delta", "pulse-delta", "differential"] = "normal",
        reset: bool = False,
        sense_type: Literal["curr", "volt"] = "curr",
    ) -> None:
        """
        set up the Keithley 6221 instruments, overwrite the specific settings here, other settings will all be
        reserved. Note that the waveform will not begin here
        """
        if reset:
            offset = 0
            source_auto_range = True
            low_grounded = True
            wave_function = "sine"

        source_6221 = self.meter
        # first must close the output to do setup
        self.output_switch("off")
        source_6221.clear()
        if reset:
            source_6221.write("*RST")
        if mea_mode == "normal":
            logger.validate(
                function == "source",
                "6221 is a source meter, so the function should be source",
            )
        if mea_mode == "delta":
            logger.info(
                "delta mode is selected, please set the specific parameters using delta_setup method"
            )
            self.delta_setup()
            mode = "dc"
        self.mea_mode = mea_mode
        if mode == "ac":
            self.info_dict["ac_dc"] = "ac"
            if wave_function is not None:
                source_6221.waveform_function = wave_function
                self.info_dict.update({"wave_function": wave_function})
            source_6221.waveform_amplitude = 0
            if offset is not None:
                source_6221.waveform_offset = offset
                self.info_dict.update({"wave_offset": offset})
            source_6221.waveform_ranging = "best"
            source_6221.waveform_use_phasemarker = True
            source_6221.waveform_phasemarker_line = 3
            source_6221.waveform_duration_set_infinity()
            source_6221.waveform_phasemarker_phase = 0
            self.info_dict.update(
                {
                    "wave_phasemarker": 0,
                }
            )
        elif mode == "dc":
            self.info_dict["ac_dc"] = "dc"
        if source_auto_range is not None:
            source_6221.source_auto_range = source_auto_range
        if low_grounded is not None:
            source_6221.output_low_grounded = low_grounded
            self.info_dict.update({"low_grounded": low_grounded})

    def delta_setup(
        self,
        *,
        delta_unit: Literal["V", "Ohms", "W", "Siemens"] = "V",
        delta_delay=0.02,
        delta_cycles: int | Literal["INF"] = "INF",
        delta_mea_sets: int | Literal["INF"] = 1,
        delta_compliance_abort: bool = True,
        delta_cold_switch: bool = False,
        trace_pts: int = 10,
    ):
        """
        set the specific parameters for delta mode
        """
        self.mea_mode = "delta"
        self.meter.delta_unit = delta_unit
        self.meter.delta_buffer_points = trace_pts

        self.meter.delta_delay = delta_delay
        self.meter.delta_cycles = delta_cycles
        self.meter.delta_measurement_sets = delta_mea_sets
        self.meter.delta_compliance_abort = delta_compliance_abort
        self.meter.delta_cold_switch = delta_cold_switch

    def output_switch(self, switch: bool | Literal["on", "off", "ON", "OFF"]):
        """
        switch the output on or off (not suitable for special modes)
        """
        switch = SWITCH_DICT.get(switch, False) if isinstance(switch, str) else switch

        if self.info_dict["output_status"] == switch:
            return

        if switch:
            if self.info_dict["ac_dc"] == "ac":
                self.meter.waveform_arm()
                self.meter.waveform_start()
                self.info_dict["output_status"] = True
            elif self.info_dict["ac_dc"] == "dc":
                self.meter.enable_source()
                self.info_dict["output_status"] = True
        else:
            if self.info_dict["ac_dc"] == "ac":
                self.meter.waveform_abort()
            elif self.info_dict["ac_dc"] == "dc":
                self.meter.disable_source()

            self.meter.waveform_amplitude = 0
            self.meter.source_current = 0
            self.info_dict["output_status"] = False

    def get_output_status(self) -> tuple[float, float, float]:
        """
        return the output value from device and also the target value set by output methods (not suitable for special modes)

        Returns:
            tuple[float, float, float]: the output value (rms for ac) and the target value
        """
        if self.info_dict["ac_dc"] == "ac":
            # amplitude for 6221 is peak to peak
            return (
                self.meter.waveform_amplitude / np.sqrt(2),
                self.output_target,
                self.meter.source_range,
            )
        elif self.info_dict["ac_dc"] == "dc":
            return (
                self.meter.source_current,
                self.output_target,
                self.meter.source_range,
            )
        else:
            raise ValueError("ac_dc term in info_dict should be either ac or dc")

    def uni_output(
        self,
        value: float | str,
        *,
        freq: Optional[float | str] = None,
        compliance: Optional[float | str] = None,
        type_str: Literal["curr", "volt"] = "curr",
        fix_range: float | str | None = None,
        alter_range: bool = False,
    ) -> float:
        # judge if the output exceeds the range first
        # since 6221 use the same source_range for both ac and dc
        # so the range could be treated in this unified method
        if self.mea_mode == "normal":
            value = convert_unit(value, "")[0]
            if abs(value) > 0.105:
                raise ValueError("6221 output should be less than 0.105A")
            range_curr = self.meter.source_range
            if (
                abs(range_curr) * 1.05 <= abs(value)
                or abs(value) <= abs(range_curr) / 100
            ) and value != 0:
                if freq is not None:
                    self.output_switch(
                        "off"
                    )  # turn off the output before changing the range for ac mode
                if (
                    fix_range is not None
                    and self.info_dict["source_range_set"] != fix_range
                ):
                    fix_range = convert_unit(fix_range, "")[0]
                    self.source_range = fix_range
                    self.info_dict["source_range_set"] = fix_range
                elif alter_range:
                    self.source_range = value
                else:
                    logger.warning("6221 output range is not suitable")
            # directly call corresponding output method if desired output type is matched
            # call setup first if desired output type is not matched
            if self.info_dict["ac_dc"] == "ac":
                if freq is not None:
                    self.rms_output(
                        value, freq=freq, compliance=compliance, type_str=type_str
                    )
                else:
                    self.setup("source", "dc")
                    self.dc_output(value, compliance=compliance, type_str=type_str)
            elif self.info_dict["ac_dc"] == "dc":
                if freq is None:
                    self.dc_output(value, compliance=compliance, type_str=type_str)
                elif freq is not None:
                    self.setup("source", "ac")
                    self.rms_output(
                        value, freq=freq, compliance=compliance, type_str=type_str
                    )

            self.output_target = convert_unit(value, "A")[0]
            return self.get_output_status()[0]
        elif self.mea_mode == "delta":
            self.meter.delta_high_source = value
            if compliance is not None:
                compliance = convert_unit(compliance, "")[0]
                self.meter.source_compliance = compliance
            self.meter.delta_arm()
            time.sleep(2)  # wait for the delta mode to be armed
            self.meter.delta_start()
            return self.meter.delta_high_source

    def rms_output(
        self,
        value: float | str,
        *,
        freq: Optional[float | str] = None,
        compliance: Optional[float | str] = None,
        type_str: Literal["curr"] = "curr",
    ):
        """
        6221 is a current source, so the output is always current
        set the output to a certain value
        if current config is dc, then call setup to reset to ac default settings
        set config manually before calling this method if special params are needed
        """
        assert type_str == "curr", (
            "6221 is a current source, so the output is always current"
        )

        if self.info_dict["ac_dc"] == "dc":
            self.setup("source", "ac")

        value = convert_unit(value, "")[0]
        # create a shortcut for turning output to 0
        if value == 0:
            self.meter.waveform_amplitude = 0
            if not self.info_dict["output_status"]:
                self.output_switch("on")
            return
        value_p2p = value * np.sqrt(2)
        if freq is not None:
            self.meter.waveform_frequency = convert_unit(freq, "Hz")[0]
            self.info_dict["frequency"] = freq

        if not self.info_dict["output_status"]:
            if compliance is not None:
                compliance = convert_unit(compliance, "")[0]
                self.meter.source_compliance = compliance
            self.meter.waveform_amplitude = value_p2p
            self.output_switch("on")

        else:
            self.meter.waveform_amplitude = value_p2p

    def dc_output(
        self,
        value: float | str,
        *,
        compliance: Optional[float | str] = None,
        type_str: Literal["curr"] = "curr",
    ):
        """
        6221 is a current source, so the output is always current
        set the output to a certain value
        """
        assert type_str == "curr", (
            "6221 is a current source, so the output is always current"
        )

        value = convert_unit(value, "")[0]
        # create a shortcut for turning output to 0
        if value == 0:
            self.meter.source_current = 0
            self.output_switch("on")
            return

        if compliance is not None:
            compliance = convert_unit(compliance, "")[0]
            self.meter.source_compliance = compliance
        self.meter.source_current = value
        self.output_switch("on")

    def sense(self, type_str: Literal["volt"] = "volt"):
        if self.mea_mode == "normal":
            logger.info("6221 is a source meter, no sense function")
        elif self.mea_mode == "delta":
            return self.meter.delta_sense

    def shutdown(self):
        if self.info_dict["output_status"]:
            self.output_switch("off")
        self.meter.shutdown()


class Wrapper2182(Meter):
    """
    Flow:
    Wrapper2182(GPIB)
    setup(channel)
    sense()
    """

    def __init__(self, GPIB: str = "GPIB0::7::INSTR"):
        super().__init__()
        self.meter = Keithley2182(GPIB, read_termination="\n")
        self.setup()
        self.info_dict = {"GPIB": GPIB, "channel": 1, "sense_type": "volt"}

    def setup(
        self,
        function: Literal["sense"] = "sense",
        *,
        channel: Literal[0, 1, 2] = 1,
        reset: bool = False,
        sense_type: Literal["volt"] = "volt",
    ) -> None:
        if reset:
            self.meter.reset()
        self.meter.active_channel = channel
        self.meter.channel_function = "voltage"
        # source_2182.sample_continuously()
        # source_2182.ch_1.voltage_offset_enabled = True
        # source_2182.ch_1.acquire_voltage_reference()
        self.meter.ch_1.setup_voltage(auto_range=True, nplc=10)
        self.meter.voltage_nplc = 10

    def info_sync(self):
        """
        no parameters to sync for 2182
        """
        pass

    @property
    def sense_range_volt(self) -> float:
        return self.meter.ch_1.voltage_range

    @sense_range_volt.setter
    def sense_range_volt(self, fix_range: float):
        self.meter.ch_1.voltage_range = fix_range

    @property
    def sense_range_curr(self) -> float:
        logger.warning("2182 no sense_range_curr")

    @sense_range_curr.setter
    def sense_range_curr(self, fix_range: float):
        logger.warning("2182 no sense_range_curr")

    def sense(self, type_str: Literal["volt"] = "volt") -> float:
        return self.meter.voltage


class Wrapper6500(Meter):
    """
    Waiting for refinement, not tested due to the limited usage of 6500
    Flow:
    Wrapper6500(GPIB)
    setup(channel)
    sense()
    """

    def __init__(self, GPIB: str = "GPIB0::16::INSTR"):
        logger.raise_error("6500 is not implemented", NotImplementedError)
        super().__init__()
        self.meter = KeithleyDMM6500(GPIB)
        self.setup("sense")
        self.info_dict = {
            "GPIB": GPIB,
            "channel": 1,
            "sense_type": "volt",
            "auto_range": True,
            "auto_zero": True,
            "terminal": "front",
        }
        logger.error("6500 wrapper not completed")

    def setup(self, function: Literal["source", "sense"], reset: bool = False) -> None:
        """default to measuring voltage"""
        if reset:
            self.meter.write("*RST")
            self.meter.nplc = 10
        self.meter.auto_range()
        if function == "sense":
            self.meter.write(":SENS:VOLT:INP AUTO")  # auto impedance
            self.meter.enable_filter("volt", "repeat", 10)
        elif function == "source":
            self.meter.autozero_enabled = True
        else:
            raise ValueError("function should be either source or sense")

    def info_sync(self):
        """
        no parameters to sync for 2182
        """
        self.info_dict.update(
            {
                "auto_range": self.meter.auto_range_status(),
                "sense_type": self.meter.mode,
                "auto_zero": self.meter.autozero_enabled,
                "terminal": self.meter.terminals_used,
            }
        )

    def sense(
        self,
        type_str: Literal["volt", "curr", "freq"] = "volt",
        max_val: Optional[float | str] = None,
        ac_dc: Literal["ac", "dc"] = "dc",
    ) -> float:
        """
        sense the voltage or current or frequency

        Args:
            type_str: "volt" or "curr" or "freq"
            max_val: the manual range for the measurement (anticipated maximum)
            ac_dc: "ac" or "dc"
        """
        if max_val is not None:
            max_val = convert_unit(max_val, "")[0]
        match type_str:
            case "volt":
                self.meter.measure_voltage(
                    max_voltage=(max_val if max_val is not None else 1),
                    ac=(ac_dc == "ac"),
                )
                self.meter.auto_range()
                return self.meter.voltage
            case "curr":
                self.meter.measure_current(
                    max_current=(max_val if max_val is not None else 1e-2),
                    ac=(ac_dc == "ac"),
                )
                self.meter.auto_range()
                return self.meter.current
            case "freq":
                self.meter.measure_frequency()
                self.meter.auto_range()
                return self.meter.frequency


class WrapperSR830(ACSourceMeter):
    def __init__(self, GPIB: str = "GPIB0::8::INSTR", reset=True):
        super().__init__()
        self.meter = SR830(GPIB)
        self.output_target = 0
        self.info_dict = {"GPIB": GPIB}
        self.safe_step = 2e-3
        self.if_source = False  # if the meter has been declared as source (as source initialization is earlier)
        self.warning_printed = False
        if reset:
            self.setup(reset=True)
        self.info_sync()

    def info_sync(self):
        self.info_dict.update(
            {
                "sensitivity": self.meter.sensitivity,
                "ref_source_trigger": self.meter.reference_source_trigger,
                "reference_source": self.meter.reference_source,
                "harmonic": self.meter.harmonic,
                "output_value": self.meter.sine_voltage,
                "output_status": self.meter.sine_voltage > 0.004,
                "frequency": self.meter.frequency,
                "filter_slope": self.meter.filter_slope,
                "time_constant": self.meter.time_constant,
                "input_config": self.meter.input_config,
                "input_coupling": self.meter.input_coupling,
                "input_grounding": self.meter.input_grounding,
                "input_notch_config": self.meter.input_notch_config,
                "reserve": self.meter.reserve,
                "filter_synchronous": self.meter.filter_synchronous,
            }
        )

    @property
    def sense_range_volt(self) -> float:
        return self.meter.sensitivity

    @sense_range_volt.setter
    def sense_range_volt(self, fix_range: float):
        self.meter.sensitivity = fix_range

    @property
    def sense_range_curr(self) -> float:
        logger.warning("confirm if in curr mode")
        return self.meter.sensitivity

    @sense_range_curr.setter
    def sense_range_curr(self, fix_range: float):
        logger.warning("confirm if in curr mode")
        self.meter.sensitivity = fix_range

    def setup(
        self,
        function: Literal["source", "sense"] = "sense",
        *,
        filter_slope=None,
        time_constant=None,
        input_config=None,
        input_coupling=None,
        input_grounding=None,
        sine_voltage=None,
        input_notch_config=None,
        reserve=None,
        filter_synchronous=None,
        reset: bool = False,
        sense_type: Literal["curr", "volt"] = "volt",
    ) -> None:
        """
        setup the SR830 instruments using pre-stored setups here, this function will not fully reset the instruments,
        only overwrite the specific settings here, other settings will all be reserved
        """
        if reset:
            self.meter.filter_slope = 24
            self.meter.time_constant = 0.3
            self.meter.input_config = "A - B"
            self.meter.input_coupling = "AC"
            self.meter.input_grounding = "Float"
            self.meter.sine_voltage = 0
            self.meter.input_notch_config = "None"
            self.meter.reserve = "Normal"
            self.meter.filter_synchronous = False
            return
        if function == "sense":
            if filter_slope is not None:
                self.meter.filter_slope = filter_slope
            if time_constant is not None:
                self.meter.time_constant = time_constant
            if input_config is not None:
                self.meter.input_config = input_config
            if input_coupling is not None:
                self.meter.input_coupling = input_coupling
            if input_grounding is not None:
                self.meter.input_grounding = input_grounding
            if input_notch_config is not None:
                self.meter.input_notch_config = input_notch_config
            if reserve is not None:
                self.meter.reserve = reserve
            if filter_synchronous is not None:
                self.meter.filter_synchronous = filter_synchronous

            if not self.if_source:
                self.meter.reference_source = "External"
            else:
                self.if_source = False  # restore the if_source to False for the next initialization, would cause unexpected behavior if called twice in one measurement
            self.info_sync()

        elif function == "source":
            if sine_voltage is not None:
                self.meter.sine_voltage = sine_voltage
            self.meter.reference_source = "Internal"
            self.if_source = True
            self.info_sync()
        else:
            raise ValueError("function should be either source or sense")

    def reference_set(
        self,
        *,
        freq: Optional[float | str] = None,
        source: Optional[Literal["Internal", "External"]] = None,
        trigger: Optional[Literal["SINE", "POS EDGE", "NEG EDGE"]] = None,
        harmonic: Optional[int] = None,
    ):
        """
        set the reference frequency and source
        """
        if freq is not None:
            self.meter.frequency = convert_unit(freq, "Hz")[0]
        if source is not None:
            self.meter.reference_source = source
        if trigger is not None:
            self.meter.reference_source_trigger = trigger
        if harmonic is not None:
            self.meter.harmonic = harmonic
        self.info_sync()

    def sense(self, type_str: Literal["volt", "curr"] = "volt") -> list:
        return self.meter.snap("X", "Y", "R", "THETA")

    def get_output_status(self) -> tuple[float, float]:
        """
        return the output value from device and also the target value set by output methods

        Returns:
            tuple[float, float]: the output value and the target value
        """
        return self.meter.sine_voltage, self.output_target

    def output_switch(self, switch: bool | Literal["on", "off", "ON", "OFF"]):
        switch = SWITCH_DICT.get(switch, False) if isinstance(switch, str) else switch
        if switch:
            # no actual switch of SR830
            self.info_dict["output_status"] = True
        else:
            self.meter.sine_voltage = 0
            self.info_dict["output_status"] = False

    def uni_output(
        self,
        value: float | str,
        *,
        freq: Optional[float | str] = None,
        compliance: Optional[float | str] = None,
        type_str: Literal["volt"] = "volt",
        fix_range: Optional[float | str] = None,
        alter_range: bool = False,
    ) -> float:
        """fix_range is not used for sr830"""
        if value > 5:
            logger.warning("exceed SR830 max output")
        self.rms_output(
            value,
            freq=freq,
            compliance=compliance,
            type_str=type_str,
            alter_range=alter_range,
        )
        self.output_target = convert_unit(value, "V")[0]
        return self.get_output_status()[0]

    def rms_output(
        self,
        value: float | str,
        *,
        freq: Optional[float | str] = None,
        compliance: Optional[float | str] = None,
        type_str: Literal["volt"] = "volt",
        alter_range: bool = False,
    ):
        if not self.warning_printed:
            logger.warning("compliance does not works for SR830")
            self.warning_printed = True

        assert type_str == "volt", (
            "SR830 is a voltage source, so the output is always voltage"
        )
        value = convert_unit(value, "V")[0]
        self.meter.sine_voltage = value
        self.info_dict["output_value"] = value
        self.info_dict["output_status"] = True
        if freq is not None and freq != self.info_dict["frequency"]:
            self.meter.frequency = convert_unit(freq, "Hz")[0]
            self.info_dict["frequency"] = freq

    def shutdown(self):
        if self.info_dict["output_status"]:
            self.output_switch("off")


class WrapperSR860(ACSourceMeter):
    def __init__(self, GPIB: str = "GPIB0::8::INSTR", reset=True):
        super().__init__()
        self.meter = SR860(GPIB)
        self.output_target = 0
        self.info_dict = {"GPIB": GPIB}
        self.safe_step = 2e-3
        self.if_source = False  # if the meter has been declared as source (as source initialization is earlier)
        self.warning_printed = False
        if reset:
            self.setup(reset=True)
        self.info_sync()

    def info_sync(self):
        self.info_dict.update(
            {
                "sensitivity": self.meter.sensitivity,
                "ref_source_trigger": self.meter.reference_source_trigger,
                "reference_source": self.meter.reference_source,
                "harmonic": self.meter.harmonic,
                "output_value": self.meter.sine_voltage,
                "output_status": self.meter.sine_voltage > 0.004,
                "frequency": self.meter.frequency,
                "filter_slope": self.meter.filter_slope,
                "time_constant": self.meter.time_constant,
                "input_config": self.meter.input_config,
                "input_coupling": self.meter.input_coupling,
                "input_grounding": self.meter.input_shields,
                "filter_synchronous": self.meter.filter_synchronous,
            }
        )

    def setup(
        self,
        function: Literal["source", "sense"] = "sense",
        *,
        filter_slope=None,
        time_constant=None,
        input_config=None,
        input_coupling=None,
        input_grounding=None,
        sine_voltage=None,
        filter_synchronous=None,
        reset: bool = False,
        sense_type: Literal["curr", "volt"] = "volt",
    ) -> None:
        """
        setup the SR830 instruments using pre-stored setups here, this function will not fully reset the instruments,
        only overwrite the specific settings here, other settings will all be reserved
        """
        if reset:
            self.meter.filter_slope = 3
            self.meter.time_constant = 0.3
            self.meter.input_config = "A-B"
            self.meter.input_coupling = "AC"
            self.meter.input_grounding = "Float"
            self.meter.sine_voltage = 0
            self.meter.filter_synchronous = False
            return
        if function == "sense":
            if filter_slope is not None:
                self.meter.filter_slope = filter_slope
            if time_constant is not None:
                self.meter.time_constant = time_constant
            if input_config is not None:
                self.meter.input_config = input_config
            if input_coupling is not None:
                self.meter.input_coupling = input_coupling
            if input_grounding is not None:
                self.meter.input_shields = input_grounding
            if filter_synchronous is not None:
                self.meter.filter_synchronous = filter_synchronous
            if not self.if_source:
                self.meter.reference_source = "EXT"
            else:
                self.if_source = False  # restore the if_source to False for the next initialization, would cause unexpected behavior if called twice in one measurement
            self.info_sync()
        elif function == "source":
            if sine_voltage is not None:
                self.meter.sine_voltage = sine_voltage
            self.meter.reference_source = "INT"
            self.if_source = True
            self.info_sync()
        else:
            raise ValueError("function should be either source or sense")

    def reference_set(
        self,
        *,
        freq: Optional[float | str] = None,
        source: Optional[Literal["Internal", "External"]] = None,
        trigger: Optional[Literal["SINE", "POS EDGE", "NEG EDGE"]] = None,
        harmonic: Optional[int] = None,
    ):
        """
        set the reference frequency and source
        """
        if freq is not None:
            self.meter.frequency = convert_unit(freq, "Hz")[0]
        if source is not None:
            self.meter.reference_source = source
        if trigger is not None:
            self.meter.reference_source_trigger = trigger
        if harmonic is not None:
            self.meter.harmonic = harmonic
        self.info_sync()

    @property
    def sense_range_volt(self) -> float:
        return self.meter.sensitivity

    @sense_range_volt.setter
    def sense_range_volt(self, fix_range: float):
        self.meter.sensitivity = fix_range

    @property
    def sense_range_curr(self) -> float:
        logger.warning("confirm if in current mode")
        return self.meter.sensitivity

    @sense_range_curr.setter
    def sense_range_curr(self, fix_range: float):
        logger.warning("confirm if in current mode")
        self.meter.sensitivity = fix_range

    def sense(
        self, type_str: Literal["volt", "curr"] = "volt"
    ) -> tuple[float, float, float, float]:
        """snap X Y and THETA from the meter and calculate the R, for compatibility with the SR830"""
        x, y, r, theta = self.meter.snap_all()
        return x, y, r, theta

    def get_output_status(self) -> tuple[float, float]:
        """
        return the output value from device and also the target value set by output methods

        Returns:
            tuple[float, float]: the output value and the target value
        """
        return self.meter.sine_voltage, self.output_target

    def output_switch(self, switch: bool | Literal["on", "off", "ON", "OFF"]):
        switch = SWITCH_DICT.get(switch, False) if isinstance(switch, str) else switch
        if switch:
            # no actual switch of SR830
            self.info_dict["output_status"] = True
        else:
            self.meter.sine_voltage = 0
            self.info_dict["output_status"] = False

    def uni_output(
        self,
        value: float | str,
        *,
        freq: Optional[float | str] = None,
        compliance: Optional[float | str] = None,
        type_str: Literal["volt"] = "volt",
        fix_range: Optional[float | str] = None,
        alter_range: bool = False,
    ) -> float:
        """fix_range is not used for sr830"""
        if value > 2:
            logger.warning("exceed SR860 max output")
        self.rms_output(
            value,
            freq=freq,
            compliance=compliance,
            type_str=type_str,
            alter_range=alter_range,
        )
        self.output_target = convert_unit(value, "V")[0]
        return self.get_output_status()[0]

    def rms_output(
        self,
        value: float | str,
        *,
        freq: Optional[float | str] = None,
        compliance: Optional[float | str] = None,
        type_str: Literal["volt"] = "volt",
        alter_range: bool = False,
    ):
        if not self.warning_printed:
            logger.warning("compliance does not works for SR860")
            self.warning_printed = True
        assert type_str == "volt", (
            "SR830 is a voltage source, so the output is always voltage"
        )
        value = convert_unit(value, "V")[0]
        self.meter.sine_voltage = value
        self.info_dict["output_value"] = value
        self.info_dict["output_status"] = True
        if freq is not None and freq != self.info_dict["frequency"]:
            self.meter.frequency = convert_unit(freq, "Hz")[0]
            self.info_dict["frequency"] = freq

    def shutdown(self):
        if self.info_dict["output_status"]:
            self.output_switch("off")


class Wrapper6430(DCSourceMeter):
    def __init__(self, GPIB: str = "GPIB0::26::INSTR"):
        super().__init__()
        self.meter: Keithley_6430
        self.meter = Keithley_6430("Keithley6430", GPIB)
        self.info_dict = {}
        self.output_target = 0
        self.safe_step = {"volt": 2e-1, "curr": 5e-6}
        self.info_sync()

    def info_sync(self):
        self.info_dict.update(
            {
                "output_status": self.meter.output_enabled(),
                "output_type": self.meter.source_mode()
                .lower()
                .replace("current", "curr")
                .replace("voltage", "volt"),
                "curr_compliance": self.meter.source_current_compliance(),
                "volt_compliance": self.meter.source_voltage_compliance(),
                "source_curr_range": self.meter.source_current_range(),
                "source_volt_range": self.meter.source_voltage_range(),
                "source_delay": self.meter.source_delay(),
                "sense_type": self.meter.sense_mode().lower(),
                "sense_auto_range": self.meter.sense_autorange(),
                "sense_curr_range": self.meter.sense_current_range(),
                "sense_volt_range": self.meter.sense_voltage_range(),
                "sense_resist_range": self.meter.sense_resistance_range(),
                "sense_resist_offset_comp": self.meter.sense_resistance_offset_comp_enabled(),
                "autozero": self.meter.autozero(),
                "source_range_set": 0,
                "sense_range_set": 0,
            }
        )

    def setup(
        self,
        function: Literal["sense", "source"] = "sense",
        *,
        auto_zero: str = "on",
        reset: bool = False,
        sense_type: Literal["curr", "volt"] = "volt",
    ):
        if function == "source":
            if reset:
                self.meter.reset()
                self.meter.output_enabled(False)
                self.meter.nplc(10)
            self.meter.autozero(auto_zero)
        elif function == "sense":
            if reset:
                self.meter.reset()
                self.meter.sense_autorange(True)
            self.meter.autozero(auto_zero)
            if sense_type == "curr":
                self.meter.sense_mode("CURR:DC")
            elif sense_type == "volt":
                self.meter.sense_mode("VOLT:DC")
        else:
            raise ValueError("function should be either source or sense")
        self.info_sync()

    @property
    def sense_range_curr(self) -> float:
        return self.meter.sense_current_range()

    @sense_range_curr.setter
    def sense_range_curr(self, fix_range: float):
        self.meter.sense_current_range(fix_range)

    @property
    def sense_range_volt(self) -> float:
        return self.meter.sense_voltage_range()

    @sense_range_volt.setter
    def sense_range_volt(self, fix_range: float):
        self.meter.sense_voltage_range(fix_range)

    @property
    def source_range(self) -> float:
        if self.info_dict["output_type"] == "curr":
            return self.meter.source_current_range()
        elif self.info_dict["output_type"] == "volt":
            return self.meter.source_voltage_range()

    @source_range.setter
    def source_range(self, fix_range: float):
        if self.info_dict["source_range_set"] != fix_range:
            if self.info_dict["output_type"] == "curr":
                self.meter.source_current_range(fix_range)
            elif self.info_dict["output_type"] == "volt":
                self.meter.source_voltage_range(fix_range)
            self.info_dict["source_range_set"] = fix_range

    @property
    def compliance(self) -> float:
        if self.info_dict["output_type"] == "curr":
            return self.meter.source_current_compliance()
        elif self.info_dict["output_type"] == "volt":
            return self.meter.source_voltage_compliance()

    @compliance.setter
    def compliance(self, compliance: float):
        if self.info_dict["source_compliance_set"] != compliance:
            if self.info_dict["output_type"] == "curr":
                self.meter.source_current_compliance(compliance)
            elif self.info_dict["output_type"] == "volt":
                self.meter.source_voltage_compliance(compliance)
            self.info_dict["source_compliance_set"] = compliance

    def sense(self, type_str: Literal["curr", "volt", "resist"]) -> float:
        if self.info_dict["output_status"] == False:
            self.output_switch("on")

        if type_str == "curr":
            if self.info_dict["sense_type"] != "curr":
                self.meter.sense_mode("CURR:DC")
                self.info_sync()
            return self.meter.sense_current()
        elif type_str == "volt":
            if self.info_dict["sense_type"] != "volt":
                self.meter.sense_mode("VOLT:DC")
                self.info_sync()
            return self.meter.sense_voltage()
        elif type_str == "resist":
            if self.info_dict["sense_type"] != "resist":
                self.meter.sense_mode("RES")
                self.info_sync()
            return self.meter.sense_resistance()

    def output_switch(self, switch: bool | Literal["on", "off", "ON", "OFF"]):
        switch = SWITCH_DICT.get(switch, False) if isinstance(switch, str) else switch
        if not switch:
            self.uni_output(0, type_str=self.info_dict["output_type"])
        self.meter.output_enabled(switch)
        self.info_dict["output_status"] = switch

    def get_output_status(self) -> tuple[float, float, float]:
        """
        return the output value from device and also the target value set by output methods

        Returns:
            tuple[float, float]: the output value and the target value
        """
        if self.meter.source_mode().lower() == "curr":
            return (
                self.meter.source_current(),
                self.output_target,
                self.meter.source_current_range(),
            )
        elif self.meter.source_mode().lower() == "volt":
            return (
                self.meter.source_voltage(),
                self.output_target,
                self.meter.source_voltage_range(),
            )

    def uni_output(
        self,
        value: float | str,
        *,
        freq=None,
        fix_range: Optional[float | str] = None,
        compliance: Optional[float | str] = None,
        type_str: Literal["curr", "volt"],
        alter_range: bool = False,
    ) -> float:
        self.dc_output(
            value,
            compliance=compliance,
            type_str=type_str,
            fix_range=fix_range,
            alter_range=alter_range,
        )
        self.output_target = convert_unit(value, "")[0]
        return self.get_output_status()[0]

    def dc_output(
        self,
        value: float | str,
        *,
        compliance: Optional[float | str] = None,
        fix_range: Optional[float | str] = None,
        type_str: Literal["curr", "volt"],
        alter_range: bool = False,
    ):
        value = convert_unit(value, "")[0]
        if fix_range is not None:
            fix_range = convert_unit(fix_range, "")[0]
        if compliance is not None:
            compliance = convert_unit(compliance, "")[0]
        # add shortcut for zero output (no need to care about output type, just set current output to 0)
        if value == 0:
            if self.info_dict["output_type"] == "curr":
                self.meter.source_current(0)
            elif self.info_dict["output_type"] == "volt":
                self.meter.source_voltage(0)
            self.output_switch("on")
            return
        # close and reopen the source meter to avoid error when switching source type
        if self.info_dict["output_type"] != type_str:
            self.output_switch("off")
            self.meter.source_mode(type_str.upper())

        if type_str == "curr":
            if (
                fix_range is not None
                and self.info_dict["source_range_set"] != fix_range
            ):
                self.meter.source_current_range(fix_range)
                self.info_dict["source_range_set"] = fix_range
                time.sleep(0.5)
            elif alter_range:
                if (
                    abs(value) <= self.meter.source_current_range() / 100
                    or abs(value) >= self.meter.source_current_range()
                ):
                    new_range = abs(value) if abs(value) > 1e-12 else 1e-12
                    self.meter.source_current_range(new_range)
                    self.info_dict["source_range_set"] = new_range
                    time.sleep(0.5)
            if (
                compliance is not None
                and compliance != self.meter.source_voltage_compliance()
            ):
                # self.meter.sense_voltage_range(convert_unit(compliance, "V")[0])
                # self.meter.sense_autorange(True)
                self.meter.source_voltage_compliance(compliance)
            self.meter.source_current(value)

        elif type_str == "volt":
            if (
                fix_range is not None
                and self.info_dict["source_range_set"] != fix_range
            ):
                self.meter.source_voltage_range(fix_range)
                self.info_dict["source_range_set"] = fix_range
                time.sleep(0.5)
            elif alter_range:
                if (
                    abs(value) <= self.meter.source_voltage_range() / 100
                    or abs(value) >= self.meter.source_voltage_range()
                ):
                    new_range = abs(value) if abs(value) > 0.2 else 0.2
                    self.meter.source_voltage_range(new_range)
                    self.info_dict["source_range_set"] = new_range
                    time.sleep(0.5)
            if (
                compliance is not None
                and compliance != self.meter.source_current_compliance()
            ):
                # self.meter.sense_current_range(convert_unit(compliance, "A")[0])
                # self.meter.sense_autorange(True)
                self.meter.source_current_compliance(compliance)
            self.meter.source_voltage(value)

        self.info_dict["output_type"] = type_str
        self.output_switch("on")

    def shutdown(self):
        self.output_switch("off")


class WrapperB2902Bchannel(DCSourceMeter):
    def __init__(self, GPIB: str = "GPIB0::25::INSTR", channel: int | str = 1):
        super().__init__()
        self.meter_all = find_or_create_instrument(
            Keysight_B2902B, "KeysightB2902B", address=GPIB
        )
        self.meter: KeysightB2902BChannel
        self.meter = self.meter_all.ch1 if int(channel) == 1 else self.meter_all.ch2
        self.info_dict = {}
        self.output_target = 0
        self.safe_step = {"volt": 1e-2, "curr": 2e-6}
        self.info_sync()

    def info_sync(self):
        self.info_dict.update(
            {
                "output_status": self.meter.output(),
                "output_type": self.meter.source_mode()
                .lower()
                .replace("current", "curr")
                .replace("voltage", "volt"),
                "curr_compliance": self.meter.source_current_compliance(),
                "volt_compliance": self.meter.source_voltage_compliance(),
                "source_curr_range": self.meter.source_current_range(),
                "source_volt_range": self.meter.source_voltage_range(),
                "sense_curr_autorange": self.meter.sense_current_autorange(),
                "sense_volt_autorange": self.meter.sense_voltage_autorange(),
                "sense_resist_autorange": self.meter.sense_resistance_autorange(),
                "sense_curr_range": self.meter.sense_current_range(),
                "sense_volt_range": self.meter.sense_voltage_range(),
                "sense_resist_range": self.meter.sense_resistance_range(),
                "source_range_set": 0,
                "source_compliance_set": 0,
                "sense_range_set": 0,
            }
        )

    def setup(
        self,
        function: Literal["sense", "source"] = "sense",
        reset: bool = False,
        sense_type: Literal["curr", "volt"] = "volt",
    ):
        if reset:
            self.meter_all.reset()
            self.meter_all.ch1.nplc(10)
            self.meter_all.ch2.nplc(10)
            self.meter_all.ch1.write("*ESE 60; *SRE 48; *CLS;")
            self.meter_all.ch2.write("*ESE 60; *SRE 48; *CLS;")

        if function == "sense":
            if reset:
                self.meter.sense_current_autorange(True)
                self.meter.sense_voltage_autorange(True)
                self.meter.sense_resistance_autorange(True)
        elif function == "source":
            if reset:
                self.meter.output(False)
                self.meter.source_current_autorange(True)
                self.meter.source_voltage_autorange(True)
        self.info_sync()

    @property
    def four_wire(self) -> bool:
        return self.meter.remote_sensing()

    @four_wire.setter
    def four_wire(self, four_wire: bool):
        self.meter.remote_sensing(four_wire)

    @property
    def sense_range_curr(self) -> float:
        return self.meter.sense_current_range()

    @sense_range_curr.setter
    def sense_range_curr(self, fix_range: float):
        self.meter.sense_current_range(fix_range)

    @property
    def sense_range_volt(self) -> float:
        return self.meter.sense_voltage_range()

    @sense_range_volt.setter
    def sense_range_volt(self, fix_range: float):
        self.meter.sense_voltage_range(fix_range)

    @property
    def source_range(self) -> float:
        if self.info_dict["output_type"] == "curr":
            return self.meter.source_current_range()
        elif self.info_dict["output_type"] == "volt":
            return self.meter.source_voltage_range()

    @source_range.setter
    def source_range(self, fix_range: float):
        if self.info_dict["source_range_set"] != fix_range:
            if self.info_dict["output_type"] == "curr":
                self.meter.source_current_range(fix_range)
            elif self.info_dict["output_type"] == "volt":
                self.meter.source_voltage_range(fix_range)
            self.info_dict["source_range_set"] = fix_range

    @property
    def compliance(self) -> float:
        if self.info_dict["output_type"] == "curr":
            return self.meter.source_current_compliance()
        elif self.info_dict["output_type"] == "volt":
            return self.meter.source_voltage_compliance()

    @compliance.setter
    def compliance(self, compliance: float):
        if self.info_dict["source_compliance_set"] != compliance:
            if self.info_dict["output_type"] == "curr":
                self.meter.source_current_compliance(compliance)
            elif self.info_dict["output_type"] == "volt":
                self.meter.source_voltage_compliance(compliance)
            self.info_dict["source_compliance_set"] = compliance

    def sense(self, type_str: Literal["curr", "volt", "resist"]) -> float:
        # if self.info_dict["output_status"] is False:
        #    self.output_switch("on")

        if type_str == "curr":
            return self.meter.sense_current()
        elif type_str == "volt":
            return self.meter.sense_voltage()
        elif type_str == "resist":
            return self.meter.sense_resistance()

    def output_switch(self, switch: bool | Literal["on", "off", "ON", "OFF"]):
        switch = SWITCH_DICT.get(switch, False) if isinstance(switch, str) else switch
        if not switch:
            if self.info_dict["output_type"] == "curr":
                self.uni_output(1e-9, type_str=self.info_dict["output_type"])
            elif self.info_dict["output_type"] == "volt":
                self.uni_output(1e-6, type_str=self.info_dict["output_type"])
            self.uni_output(0, type_str=self.info_dict["output_type"])
        self.meter.output(switch)
        self.info_dict["output_status"] = switch

    def get_output_status(self) -> tuple[float, float, float]:
        """
        return the output value from device and also the target value set by output methods

        Returns:
            tuple[float, float]: the output value and the target value
        """
        if self.meter.source_mode().lower() == "curr":
            return (
                self.meter.source_current(),
                self.output_target,
                self.meter.source_current_range(),
            )
        elif self.meter.source_mode().lower() == "volt":
            return (
                self.meter.source_voltage(),
                self.output_target,
                self.meter.source_voltage_range(),
            )

    def uni_output(
        self,
        value: float | str,
        *,
        freq=None,
        fix_range: Optional[float | str] = None,
        compliance: Optional[float | str] = None,
        type_str: Literal["curr", "volt"],
        alter_range: bool = False,
    ) -> float:
        self.dc_output(
            value,
            compliance=compliance,
            type_str=type_str,
            fix_range=fix_range,
            alter_range=alter_range,
        )
        self.output_target = convert_unit(value, "")[0]
        return self.get_output_status()[0]

    def dc_output(
        self,
        value: float | str,
        *,
        compliance: Optional[float | str] = None,
        fix_range: Optional[float | str] = None,
        type_str: Literal["curr", "volt"],
        alter_range: bool = False,
    ):
        value = convert_unit(value, "")[0]
        if fix_range is not None:
            fix_range = convert_unit(fix_range, "")[0]
        if compliance is not None:
            compliance = convert_unit(compliance, "")[0]
        # add shortcut for zero output (no need to care about output type, just set current output to 0)
        if value == 0:
            if self.info_dict["output_type"] == "curr":
                self.meter.source_current(0)
            elif self.info_dict["output_type"] == "volt":
                self.meter.source_voltage(0)
            self.output_switch("on")
            return
        # close and reopen the source meter to avoid error when switching source type
        if self.info_dict["output_type"] != type_str:
            self.output_switch("off")
            self.meter.source_mode(type_str.upper())

        if type_str == "curr":
            if (
                fix_range is not None
                and self.info_dict["source_range_set"] != fix_range
            ):
                self.meter.source_current_range(fix_range)
                self.info_dict["source_range_set"] = fix_range
                time.sleep(0.5)
            elif alter_range:
                if (
                    abs(value) <= self.meter.source_current_range() / 100
                    or abs(value) >= self.meter.source_current_range()
                ):
                    new_range = abs(value) if abs(value) > 1e-12 else 1e-12
                    self.meter.source_current_range(new_range)
                    self.info_dict["source_range_set"] = new_range
                    time.sleep(0.5)
            if (
                compliance is not None
                and compliance != self.meter.source_voltage_compliance()
            ):
                # self.meter.sense_voltage_range(convert_unit(compliance, "V")[0])
                # self.meter.sense_voltage_autorange(True)
                self.meter.source_voltage_compliance(compliance)
            self.meter.source_current(value)

        elif type_str == "volt":
            if (
                fix_range is not None
                and self.info_dict["source_range_set"] != fix_range
            ):
                self.meter.source_voltage_range(fix_range)
                self.info_dict["source_range_set"] = fix_range
                time.sleep(0.5)
            elif alter_range:
                if (
                    abs(value) <= self.meter.source_voltage_range() / 100
                    or abs(value) >= self.meter.source_voltage_range()
                ):
                    new_range = abs(value) if abs(value) > 0.2 else 0.2
                    self.meter.source_voltage_range(new_range)
                    self.info_dict["source_range_set"] = new_range
                    time.sleep(0.5)
            if (
                compliance is not None
                and compliance != self.meter.source_current_compliance()
            ):
                # self.meter.sense_current_range(convert_unit(compliance, "A")[0])
                # self.meter.sense_current_autorange(True)
                self.meter.source_current_compliance(convert_unit(compliance, "A")[0])
            self.meter.source_voltage(value)

        self.info_dict["output_type"] = type_str
        self.output_switch("on")

    def shutdown(self):
        self.output_switch("off")


class Wrapper2400(DCSourceMeter):
    def __init__(self, GPIB: str = "GPIB0::24::INSTR"):
        super().__init__()
        self.meter = Keithley2400("Keithley2401", GPIB)
        self.info_dict = {}
        self.output_target = 0
        self.safe_step = {"volt": 1e-2, "curr": 2e-6}
        self.info_sync()

    def info_sync(self):
        self.info_dict.update(
            {
                "output_status": self.meter.output(),
                "output_type": self.meter.mode()
                .lower()
                .replace("current", "curr")
                .replace("voltage", "volt"),
                "curr_compliance": self.meter.compliancei(),
                "volt_compliance": self.meter.compliancev(),
                "source_curr_range": self.meter.rangei(),
                "source_volt_range": self.meter.rangev(),
                "sense_curr_range": self.meter.rangei(),
                "sense_volt_range": self.meter.rangev(),
                "sense_type": self.meter.sense().lower(),
                "source_range_set": 0,
                "source_compliance_set": 0,
                "sense_range_set": 0,
            }
        )

    def setup(
        self,
        function: Literal["sense", "source"] = "sense",
        reset: bool = False,
        sense_type: Literal["curr", "volt"] = "volt",
    ):
        if reset:  # reset will also reset the GPIB
            self.meter.write("*RST")
            self.meter.write("*CLS")
            self.meter.write(":TRAC:FEED:CONT NEV")  # disables data buffer
            self.meter.write(":RES:MODE MAN")  # disables auto resistance
            self.meter.nplci(10)
            self.meter.nplcv(10)
        self.info_sync()

    @property
    def sense_range_curr(self) -> float:
        if self.info_dict["sense_type"] == "curr":
            pass
        elif self.info_dict["sense_type"] == "volt":
            logger.warning("currently in volt sense mode")
        return self.meter.rangei()

    @sense_range_curr.setter
    def sense_range_curr(self, fix_range: float):
        if self.info_dict["sense_type"] == "curr":
            pass
        elif self.info_dict["sense_type"] == "volt":
            logger.warning("currently in volt sense mode")
        self.meter.rangei(fix_range)

    @property
    def sense_range_volt(self) -> float:
        if self.info_dict["sense_type"] == "curr":
            logger.warning("currently in curr sense mode")
        elif self.info_dict["sense_type"] == "volt":
            pass
        return self.meter.rangev()

    @sense_range_volt.setter
    def sense_range_volt(self, fix_range: float):
        if self.info_dict["sense_type"] == "curr":
            logger.warning("currently in curr sense mode")
        elif self.info_dict["sense_type"] == "volt":
            pass
        self.meter.rangev(fix_range)

    @property
    def source_range(self) -> float:
        if self.info_dict["output_type"] == "curr":
            return self.meter.rangei()
        elif self.info_dict["output_type"] == "volt":
            return self.meter.rangev()

    @source_range.setter
    def source_range(self, fix_range: float):
        if self.info_dict["source_range_set"] != fix_range:
            if self.info_dict["output_type"] == "curr":
                self.meter.rangei(fix_range)
            elif self.info_dict["output_type"] == "volt":
                self.meter.rangev(fix_range)
            self.info_dict["source_range_set"] = fix_range

    @property
    def compliance(self) -> float:
        if self.info_dict["output_type"] == "curr":
            return self.meter.compliancei()
        elif self.info_dict["output_type"] == "volt":
            return self.meter.compliancev()

    @compliance.setter
    def compliance(self, compliance: float):
        if self.info_dict["source_compliance_set"] != compliance:
            if self.info_dict["output_type"] == "curr":
                self.meter.compliancei(compliance)
            elif self.info_dict["output_type"] == "volt":
                self.meter.compliancev(compliance)
            self.info_dict["source_compliance_set"] = compliance

    def sense(self, type_str: Literal["curr", "volt", "resist"]) -> float:
        if type_str == "curr":
            if self.info_dict["output_type"] == "curr":
                logger.info("in curr mode, print the set point")
            return self.meter.curr()
        elif type_str == "volt":
            if self.info_dict["output_type"] == "volt":
                logger.info("in curr mode, print the set point")
            return self.meter.volt()
        elif type_str == "resist":
            return self.meter.resistance()

    def get_output_status(self) -> tuple[float, float, float]:
        """
        return the output value from device and also the target value set by output methods

        Returns:
            tuple[float, float, float]: the output value, target value and range
        """
        if self.meter.mode().lower() == "curr":
            if self.info_dict["output_status"] == False:
                return 0, self.output_target, self.meter.rangei()
            return self.meter.curr(), self.output_target, self.meter.rangei()
        elif self.meter.mode().lower() == "volt":
            if self.info_dict["output_status"] == False:
                return 0, self.output_target, self.meter.rangev()
            return self.meter.volt(), self.output_target, self.meter.rangev()

    def output_switch(self, switch: bool | Literal["on", "off", "ON", "OFF"]):
        switch = SWITCH_DICT.get(switch, False) if isinstance(switch, str) else switch
        if not switch:
            self.uni_output(0, type_str=self.info_dict["output_type"])
        self.meter.output(switch)
        self.info_dict["output_status"] = switch

    def uni_output(
        self,
        value: float | str,
        *,
        freq=None,
        fix_range: Optional[float | str] = None,
        compliance: Optional[float | str] = None,
        type_str: Literal["curr", "volt"],
        alter_range: bool = False,
    ) -> float:
        self.dc_output(
            value,
            compliance=compliance,
            type_str=type_str,
            fix_range=fix_range,
            alter_range=alter_range,
        )
        self.output_target = convert_unit(value, "")[0]
        return self.get_output_status()[0]

    def dc_output(
        self,
        value: float | str,
        *,
        fix_range: Optional[float | str] = None,
        compliance: Optional[float | str] = None,
        type_str: Literal["curr", "volt"],
        alter_range: bool = False,
    ):
        value = convert_unit(value, "")[0]
        if fix_range is not None:
            fix_range = convert_unit(fix_range, "")[0]
        if compliance is not None:
            compliance = convert_unit(compliance, "")[0]
        # add shortcut for zero output (no need to care about output type, just set current output to 0)
        if value == 0:
            if self.info_dict["output_type"] == "curr":
                self.meter.curr(0)
            elif self.info_dict["output_type"] == "volt":
                self.meter.volt(0)
            self.output_switch("on")
            return
        # close and reopen the source meter to avoid error when switching source type
        if self.info_dict["output_type"] != type_str:
            self.output_switch("off")
            self.meter.mode(type_str.upper())

        if type_str == "curr":
            if (
                fix_range is not None
                and self.info_dict["source_range_set"] != fix_range
            ):
                self.meter.rangei(fix_range)
                self.info_dict["source_range_set"] = fix_range
                time.sleep(0.5)
            elif alter_range:
                if (
                    abs(value) <= self.meter.rangei() / 100
                    or abs(value) >= self.meter.rangei()
                ):
                    new_range = value if abs(value) > 1e-6 else 1e-6
                    self.meter.rangei(new_range)
                    self.info_dict["source_range_set"] = new_range
                    time.sleep(0.5)
            if compliance is not None and compliance != self.meter.compliancev():
                self.meter.compliancev(compliance)
            self.meter.curr(value)

        elif type_str == "volt":
            if (
                fix_range is not None
                and self.info_dict["source_range_set"] != fix_range
            ):
                self.meter.rangev(fix_range)
                self.info_dict["source_range_set"] = fix_range
                time.sleep(0.5)
            elif alter_range:
                if (
                    abs(value) <= self.meter.rangev() / 100
                    or abs(value) >= self.meter.rangev()
                ):
                    new_range = value if abs(value) > 0.2 else 0.2
                    self.meter.rangev(new_range)
                    self.info_dict["source_range_set"] = new_range
                    time.sleep(0.5)
            if compliance is not None and compliance != self.meter.compliancei():
                self.meter.compliancei(compliance)
            self.meter.volt(value)

        self.info_dict["output_type"] = type_str
        self.output_switch("on")

    def shutdown(self):
        self.meter.curr(0)
        self.meter.volt(0)
        self.output_switch("off")


class Wrapper2450(DCSourceMeter):
    ##TODO: not tested yet
    def __init__(self, GPIB: str = "GPIB0::18::INSTR"):
        super().__init__()
        try:
            self.meter = Keithley2450("Keithley2450", GPIB)
        except:
            self.meter = Keithley2450("Keithley2450_2", GPIB)
        self.info_dict = {}
        self.output_target = 0
        self.safe_step = {"volt": 1e-2, "curr": 2e-6}
        self.info_sync()

    def info_sync(self):
        self.info_dict.update(
            {
                "output_status": self.meter.output_enabled(),
                "output_type": self.meter.source_function()
                .lower()
                .replace("current", "curr")
                .replace("voltage", "volt"),
                "compliance": self.meter.source.limit(),
                "source_range": self.meter.source.range(),
                "sense_range": self.meter.sense.range(),
                "sense_type": self.meter.sense_function()
                .lower()
                .replace("current", "curr")
                .replace("voltage", "volt")
                .replace("resistance", "resist"),
                "sense_autozero": self.meter.sense.auto_zero_enabled(),
                "source_range_set": 0,
                "source_compliance_set": 0,
                "sense_range_set": 0,
            }
        )

    def setup(
        self,
        function: Literal["sense", "source"] = "sense",
        *,
        terminal: Literal["front", "rear"] = "rear",
        reset: bool = False,
        sense_type: Literal["curr", "volt"] = "curr",
    ):
        if reset:
            self.meter.reset()
            self.meter.sense.auto_range(True)
            self.meter.terminals(terminal)
            self.meter.write("*SRE 32")
            self.meter.write("*CLS")
            self.meter.write(":SOUR:VOLT:READ:BACK ON")
            self.meter.write(":SOUR:CURR:READ:BACK ON")
            time.sleep(0.5)
            self.meter.sense.nplc(10)
        if function == "source":
            if reset:
                self.meter.source.auto_range(True)
        if function == "sense":
            self.meter.sense.function(
                sense_type.replace("curr", "current").replace("volt", "voltage")
            )
        self.info_sync()

    def set_terminal(self, terminal: Literal["front", "rear"]):
        self.meter.terminals(terminal)
        self.info_sync()

    @property
    def sense_range_curr(self) -> float:
        if self.info_dict["sense_type"] == "curr":
            pass
        elif self.info_dict["sense_type"] == "volt":
            logger.warning("currently in volt sense mode")
        return self.meter.sense.range()

    @sense_range_curr.setter
    def sense_range_curr(self, fix_range: float):
        if self.info_dict["sense_type"] == "curr":
            pass
        elif self.info_dict["sense_type"] == "volt":
            logger.warning("currently in volt sense mode")
        self.meter.sense.range(fix_range)

    @property
    def sense_range_volt(self) -> float:
        if self.info_dict["sense_type"] == "curr":
            logger.warning("currently in curr sense mode")
        elif self.info_dict["sense_type"] == "volt":
            pass
        return self.meter.sense.range()

    @sense_range_volt.setter
    def sense_range_volt(self, fix_range: float):
        if self.info_dict["sense_type"] == "curr":
            logger.warning("currently in curr sense mode")
        elif self.info_dict["sense_type"] == "volt":
            pass
        self.meter.sense.range(fix_range)

    @property
    def source_range(self) -> float:
        return self.meter.source.range()

    @source_range.setter
    def source_range(self, fix_range: float):
        if self.info_dict["source_range_set"] != fix_range:
            self.meter.source.range(fix_range)
            self.info_dict["source_range_set"] = fix_range

    @property
    def compliance(self) -> float:
        return self.meter.source.limit()

    @compliance.setter
    def compliance(self, compliance: float):
        if self.info_dict["source_compliance_set"] != compliance:
            self.meter.source.limit(compliance)
            self.info_dict["source_compliance_set"] = compliance

    def sense(self, type_str: Literal["curr", "volt", "resist"]) -> float:
        if self.info_dict["sense_type"] == type_str:
            pass
        else:
            self.meter.sense.function(
                type_str.replace("curr", "current")
                .replace("volt", "voltage")
                .replace("resist", "resistance")
            )
            self.info_sync()
        return self.meter.sense._measure()

    @property
    def four_wire(self) -> bool:
        return self.meter.sense.four_wire_measurement()

    @four_wire.setter
    def four_wire(self, four_wire: bool):
        self.output_switch("off")
        self.meter.sense.four_wire_measurement(four_wire)

    def get_output_status(self) -> tuple[float, float, float]:
        """
        return the output value from device and also the target value set by output methods

        Returns:
            tuple[float, float, float]: the real output value, target value and range
        """
        if not self.info_dict["output_status"]:
            return 0, self.output_target, self.meter.source.range()
        if self.meter.source.function() == "current":
            return (
                self.meter.source.current(),
                self.output_target,
                self.meter.source.range(),
            )
        elif self.meter.source.function() == "voltage":
            return (
                self.meter.source.voltage(),
                self.output_target,
                self.meter.source.range(),
            )

    def output_switch(
        self,
        switch: bool | Literal["on", "off", "ON", "OFF"],
        *,
        force_sync: bool = True,
    ):
        if force_sync:
            self.info_sync()
        switch = SWITCH_DICT.get(switch, False) if isinstance(switch, str) else switch
        if self.info_dict["output_status"] == switch:
            return
        if not switch:
            self.uni_output(0, type_str=self.info_dict["output_type"])
        self.meter.output_enabled(switch)
        self.info_dict["output_status"] = switch

    def uni_output(
        self,
        value: float | str,
        *,
        freq=None,
        fix_range: Optional[float | str] = None,
        compliance: float | str = None,
        type_str: Literal["curr", "volt"],
        alter_range: bool = False,
    ) -> float:
        self.dc_output(
            value,
            compliance=compliance,
            type_str=type_str,
            fix_range=fix_range,
            alter_range=alter_range,
        )
        self.output_target = convert_unit(value, "")[0]
        return self.get_output_status()[0]

    def dc_output(
        self,
        value: float | str,
        *,
        fix_range: Optional[float | str] = None,
        compliance: Optional[float | str] = None,
        type_str: Literal["curr", "volt"],
        alter_range: bool = False,
    ):
        value = convert_unit(value, "")[0]
        if fix_range is not None:
            fix_range = convert_unit(fix_range, "")[0]
        if compliance is not None:
            compliance = convert_unit(compliance, "")[0]
        # close and reopen the source meter to avoid error when switching source type
        if self.info_dict["output_type"] != type_str:
            self.output_switch("off")
            self.meter.source.function(
                type_str.replace("curr", "current").replace("volt", "voltage")
            )
            self.info_dict["output_type"] = type_str
        if alter_range and fix_range is None:
            if (
                abs(value) <= self.meter.source.range() / 100
                or abs(value) >= self.meter.source.range()
            ):
                fix_range = value
        if fix_range is not None and self.info_dict["source_range_set"] != fix_range:
            self.source_range = fix_range
            time.sleep(0.5)

        if compliance is not None and compliance != self.meter.source.limit():
            self.meter.source.limit(compliance)

        if type_str == "curr":
            self.meter.source.current(value)

        elif type_str == "volt":
            self.meter.source.voltage(value)

        self.info_dict["output_type"] = type_str
        self.output_switch("on")

    def shutdown(self):
        if self.info_dict["output_type"] == "curr":
            self.meter.source.current(0)
        elif self.info_dict["output_type"] == "volt":
            self.meter.source.voltage(0)
        self.output_switch("off")
