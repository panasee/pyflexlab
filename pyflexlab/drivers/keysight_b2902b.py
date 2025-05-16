from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal


from qcodes.instrument import (
    InstrumentBaseKWArgs,
    InstrumentChannel,
    VisaInstrument,
)

if TYPE_CHECKING:

    from typing_extensions import Unpack

from qcodes.utils.validators import Enum, Lists, Numbers
from qcodes.utils.helpers import create_on_off_val_mapping
import warnings

log = logging.getLogger(__name__)
on_off_vals = create_on_off_val_mapping(on_val=1, off_val=0)

def _ascii_float_parser(s: str) -> float:
    """
    Parse ASCII float values from the instrument.
    """
    if float(s) == 9.91E37:
        return float("nan")
    if abs(float(s)) == 9.90E37:
        return float("inf") * float(s) / abs(float(s))
    return float(s)


class KeysightB2902BChannel(InstrumentChannel):
    def __init__(
        self,
        parent: VisaInstrument,
        name: str,
        channel: int | str,
        **kwargs: Unpack[InstrumentBaseKWArgs],
    ) -> None:
        if str(channel) not in ["1", "2"]:
            raise ValueError(f"channel must be 1 or 2, got {channel}")
        super().__init__(parent, name, **kwargs)
        self.channel = str(channel)
        self.function_name_mapping = {"measurement": "ACQ", "output": "TRAN", "all": "ALL"}

        self.auto_output_on_status = self.add_parameter(
            "auto_output_on_status",
            set_cmd=f"OUTP{self.channel}:ON:AUTO {{}}",
            get_cmd=f"OUTP{self.channel}:ON:AUTO?",
            val_mapping=on_off_vals,
            docstring="Auto output on status.",
        )
        """Parameter auto_output_on_status"""

        self.auto_output_off_status = self.add_parameter(
            "auto_output_off_status",
            set_cmd=f"OUTP{self.channel}:OFF:AUTO {{}}",
            get_cmd=f"OUTP{self.channel}:OFF:AUTO?",
            val_mapping=on_off_vals,
            docstring="Auto output off status.",
        )
        """Parameter auto_output_off_status"""


        self.auto_output_off_mode = self.add_parameter(
            "auto_output_off_mode",
            set_cmd=f"OUTP{self.channel}:OFF:MODE {{}}",
            get_cmd=f"OUTP{self.channel}:OFF:MODE?",
            vals=Enum("ZERO", "HIZ", "NORM"),
            docstring="Auto output off mode.",
        )
        """Parameter auto_output_off_mode"""

        self.output_low = self.add_parameter(
            "output_low",
            set_cmd=f"OUTP{self.channel}:LOW {{}}",
            get_cmd=f"OUTP{self.channel}:LOW?",
            docstring="Output low level.",
            val_mapping={
                "float": "FLO",
                "ground": "GRO",
            },
        )
        """Parameter output_low"""

        self.compliance_off_protection = self.add_parameter(
            "compliance_off_protection",
            set_cmd=f"OUTP{self.channel}:PROT {{}}",
            get_cmd=f"OUTP{self.channel}:PROT?",
            docstring="whether to set output to 0 and disable output when the output reaches the compliance limit.",
            val_mapping=on_off_vals,
        )
        """Parameter compliance_off_protection"""

        self.output = self.add_parameter(
            "output",
            set_cmd=f"OUTP{self.channel}:STAT {{}}",
            get_cmd=f"OUTP{self.channel}:STAT?",
            docstring="Output status.",
            val_mapping=on_off_vals,
        )
        """Parameter output"""

        self.source_mode = self.add_parameter(
            "source_mode",
            set_cmd=f"SOUR{self.channel}:FUNC:MODE {{}}",
            get_cmd=f"SOUR{self.channel}:FUNC:MODE?",
            docstring="Source mode, 'VOLT' or 'CURR'.",
            vals=Enum("VOLT", "CURR"),
        )
        """Parameter source_mode"""

        self.source_shape = self.add_parameter(
            "source_shape",
            set_cmd=f"SOUR{self.channel}:FUNC:SHAP {{}}",
            get_cmd=f"SOUR{self.channel}:FUNC:SHAP?",
            docstring="Source shape, 'DC' or 'PULS'.",
            vals=Enum("DC", "PULS"),
        )
        """Parameter source_shape"""

        self.source_current = self.add_parameter(
            "source_current",
            set_cmd=f"SOUR{self.channel}:CURR {{}}",
            get_cmd=f"SOUR{self.channel}:CURR?",
            docstring="Source current.",
            unit="A",
            get_parser=float,
            vals = Numbers(-3.03, 3.03),
        )
        """Parameter source_current"""

        self.source_current_range = self.add_parameter(
            "source_current_range",
            get_cmd=f"SOUR{self.channel}:CURR:RANG?",
            set_cmd=f"SOUR{self.channel}:CURR:RANG {{}}",
            docstring="Source current range.",
            unit="A",
            get_parser=float,
            vals = Numbers(0, 3.03),
        )
        """Parameter source_current_range"""

        self.source_current_autorange = self.add_parameter(
            "source_current_autorange",
            get_cmd=f"SOUR{self.channel}:CURR:RANG:AUTO?",
            set_cmd=f"SOUR{self.channel}:CURR:RANG:AUTO {{}}",
            docstring="Source current autorange.",
            val_mapping=on_off_vals,
        )
        """Parameter source_current_autorange"""

        self.source_current_compliance = self.add_parameter(
            "source_current_compliance",
            get_cmd=f"SENS{self.channel}:CURR:PROT?",
            set_cmd=f"SENS{self.channel}:CURR:PROT {{}}",
            docstring="source current compliance",
            unit="A",
            get_parser=float,
            vals = Numbers(0, 3.03),
        )
        """Parameter source_current_compliance"""

        self.source_current_compliance_tripped = self.add_parameter(
            "source_current_compliance_tripped",
            get_cmd=f"SENS{self.channel}:CURR:PROT:TRIP?",
            docstring="source current compliance tripped",
            val_mapping=on_off_vals,
        )
        """Parameter source_current_compliance_tripped"""

        self.source_voltage = self.add_parameter(
            "source_voltage",
            set_cmd=f"SOUR{self.channel}:VOLT {{}}",
            get_cmd=f"SOUR{self.channel}:VOLT?",
            docstring="Source voltage.",
            unit="V",
            get_parser=float,
            vals = Numbers(-210, 210),
        )
        """Parameter source_voltage"""

        self.source_voltage_range = self.add_parameter(
            "source_voltage_range",
            get_cmd=f"SOUR{self.channel}:VOLT:RANG?",
            set_cmd=f"SOUR{self.channel}:VOLT:RANG {{}}",
            docstring="Source voltage range.",
            unit="V",
            get_parser=float,
            vals = Numbers(0, 210),
        )
        """Parameter source_voltage_range"""
        
        self.source_voltage_autorange = self.add_parameter(
            "source_voltage_autorange",
            get_cmd=f"SOUR{self.channel}:VOLT:RANG:AUTO?",
            set_cmd=f"SOUR{self.channel}:VOLT:RANG:AUTO {{}}",
            docstring="Source voltage autorange.",
            val_mapping=on_off_vals,
        )
        """Parameter source_voltage_autorange"""

        self.source_voltage_compliance = self.add_parameter(
            "source_voltage_compliance",
            get_cmd=f"SENS{self.channel}:VOLT:PROT?",
            set_cmd=f"SENS{self.channel}:VOLT:PROT {{}}",
            docstring="source voltage compliance",
            unit="V",
            get_parser=float,
            vals = Numbers(0, 210),
        )
        """Parameter source_voltage_compliance"""

        self.source_voltage_compliance_tripped = self.add_parameter(
            "source_voltage_compliance_tripped",
            get_cmd=f"SENS{self.channel}:VOLT:PROT:TRIP?",
            docstring="source voltage compliance tripped",
            val_mapping=on_off_vals,
        )
        """Parameter source_voltage_compliance_tripped"""

        self.source_idle_status = self.add_parameter(
            "source_idle_status",
            get_cmd=f"IDLE:TRAN?{self.channel}",
            docstring="Source idle status.",
            val_mapping={
                "idle": "1",
                "busy": "0",
            },
        )
        """Parameter source_idle_status"""

        self.sense_current = self.add_parameter(
            "sense_current",
            get_cmd=f"MEAS:CURR? (@{self.channel})",
            docstring="One-shot Sense current.",
            unit="A",
            get_parser=_ascii_float_parser,
        )
        """Parameter sense_current"""

        self.sense_voltage = self.add_parameter(
            "sense_voltage",
            get_cmd=f"MEAS:VOLT? (@{self.channel})",
            docstring="One-shot Sense voltage.",
            unit="V",
            get_parser=_ascii_float_parser,
        )
        """Parameter sense_voltage"""

        self.sense_resistance = self.add_parameter(
            "sense_resistance",
            get_cmd=f"MEAS:RES? (@{self.channel})",
            docstring="One-shot Sense resistance.",
            unit="Ohm",
            get_parser=_ascii_float_parser,
        )
        """Parameter sense_resistance"""

        self.nplc = self.add_parameter(
            "nplc",
            get_cmd=f"SENS{self.channel}:VOLT:NPLC?",
            set_cmd=f"SENS{self.channel}:VOLT:NPLC {{}}",
            docstring="the number of power line cycles (NPLC) value",
            unit="",
            get_parser=float,
            vals = Numbers(4E-4, 100),  # for 50 Hz
        )
        """Parameter nplc"""

        self.integration_time = self.add_parameter(
            "integration_time",
            get_cmd=f"SENS{self.channel}:VOLT:APER?",
            set_cmd=f"SENS{self.channel}:VOLT:APER {{}}",
            docstring="the integration time value",
            unit="s",
            get_parser=float,
            vals = Numbers(8E-6, 2),
        )
        """Parameter integration_time"""

        self.auto_integration_time = self.add_parameter(
            "auto_integration_time",
            get_cmd=f"SENS{self.channel}:VOLT:APER:AUTO?",
            set_cmd=f"SENS{self.channel}:VOLT:APER:AUTO {{}}",
            docstring="auto integration time",
            val_mapping=on_off_vals,
        )
        """Parameter auto_integration_time"""


        self.sense_current_autorange = self.add_parameter(
            "sense_current_autorange",
            get_cmd=f"SENS{self.channel}:CURR:RANG:AUTO?",
            set_cmd=f"SENS{self.channel}:CURR:RANG:AUTO {{}}",
            docstring="sense current autorange",
            val_mapping=on_off_vals,
        )
        """Parameter sense_current_autorange"""

        self.sense_voltage_autorange = self.add_parameter(
            "sense_voltage_autorange",
            get_cmd=f"SENS{self.channel}:VOLT:RANG:AUTO?",
            set_cmd=f"SENS{self.channel}:VOLT:RANG:AUTO {{}}",
            docstring="sense voltage autorange",
            val_mapping=on_off_vals,
        )
        """Parameter sense_voltage_autorange"""

        self.sense_resistance_autorange = self.add_parameter(
            "sense_resistance_autorange",
            get_cmd=f"SENS{self.channel}:RES:RANG:AUTO?",
            set_cmd=f"SENS{self.channel}:RES:RANG:AUTO {{}}",
            docstring="sense resistance autorange",
            val_mapping=on_off_vals,
        )
        """Parameter sense_resistance_autorange"""

        self.sense_current_range = self.add_parameter(
            "sense_current_range",
            get_cmd=f"SENS{self.channel}:CURR:RANG:UPP?",
            set_cmd=f"SENS{self.channel}:CURR:RANG:UPP {{}}",
            docstring="sense current range",
            unit="A",
            get_parser=float,
            vals = Numbers(0, 10),
        )
        """Parameter sense_current_range"""

        self.sense_voltage_range = self.add_parameter(
            "sense_voltage_range",
            get_cmd=f"SENS{self.channel}:VOLT:RANG:UPP?",
            set_cmd=f"SENS{self.channel}:VOLT:RANG:UPP {{}}",
            docstring="sense voltage range",
            unit="V",
            get_parser=float,
            vals = Numbers(0, 200),
        )
        """Parameter sense_voltage_range"""

        self.remote_sensing = self.add_parameter(
            "remote_sensing",
            get_cmd=f":SENS{self.channel}:REM?",
            set_cmd=f":SENS{self.channel}:REM {{}}",
            docstring="remote sensing",
            val_mapping=on_off_vals,
        )
        """Parameter remote_sensing"""

        self.sense_resistance_range = self.add_parameter(
            "sense_resistance_range",
            get_cmd=f"SENS{self.channel}:RES:RANG:UPP?",
            set_cmd=f"SENS{self.channel}:RES:RANG:UPP {{}}",
            docstring="sense resistance range",
            unit="Ohm",
            get_parser=float,
            vals = Numbers(0, 2E8),
        )
        """Parameter sense_resistance_range"""

        # followings are for array measurement(like sweep)
        self.sense_all_data = self.add_parameter(
            "sense_all_data",
            get_cmd=f"SENS{self.channel}:DATA?",
            docstring="sense all data",
            unit="",
        )
        """Parameter sense_all_data"""

    def abort(self, action: Literal["measurement", "output", "all"] = "all") -> None:
        """
        Abort the current operation.
        """
        self.write(f"ABOR:{self.function_name_mapping[action]} (@{self.channel})")
    
    def initiate_action(self, action: Literal["measurement", "output", "all"] = "all") -> None:
        """
        Initiate the current operation.
        """
        self.write(f"INIT:{self.function_name_mapping[action]} (@{self.channel})")


class Keysight_B2902B(VisaInstrument):
    r"""
    This is the Qcodes driver for the Keysight B2902B.

    Args:
        name: The name used internally by QCoDeS
        address: Network address or alias of the instrument
        terminator: Termination character in VISA communication
        reset: resets to default values
    """

    def __init__(
        self, name: str, address: str, terminator="\n", reset: bool = False, **kwargs
    ):
        super().__init__(name, address, terminator=terminator, **kwargs)
        self.channel_count = 2
        for ch in range(1, self.channel_count + 1):
            ch_name = f"ch{ch}"
            channel = KeysightB2902BChannel(self, ch_name, ch)
            self.add_submodule(ch_name, channel)  # ch1, ch2

        self.data_format = self.add_parameter(
            "data_format",
            set_cmd="FORM {}",
            get_cmd="FORM?",
            vals=Enum("ASCii", "REAL,32", "REAL,64"),
            docstring="Data output format. REAL,32/64 specifies IEEE-754 single/double precision format.",
        )
        """Parameter data_format"""

        self.data_elements = self.add_parameter(
            "data_elements",
            set_cmd="FORM:ELEM:SENS {}",
            get_cmd="FORM:ELEM:SENS?",
            vals=Lists(Enum("VOLT", "CURR", "RES", "TIME", "STAT", "SOUR")),
            docstring="the elements included in the sense or measurement result data returned by the :FETCh?, :READ?, :MEASure?, or :TRACe:DATA? command. ",
        )
        """Parameter data_elements"""

        self.beeper_state = self.add_parameter(
            "beeper_state",
            set_cmd="SYST:BEEP:STAT {}",
            get_cmd="SYST:BEEP:STAT?",
            docstring="Beeper control.",
            val_mapping=on_off_vals,
        )
        """Parameter beeper_state"""

        self.connect_message()

        if reset:
            self.reset()

    def reset(self) -> None:
        """
        Resets instrument to default values
        """
        self.write("*RST")

    def get_error_message(self) -> str:
        """
        Get the error message from the instrument and clear the error queue.
        """
        return self.ask("SYST:ERR:ALL?")

    def beep(self, frequency: float = 1000, duration: float = 1) -> None:
        """
        Generates a beep sound.

        Args:
            frequency: Frequency of the beep in Hz.
            duration: Duration of the beep in seconds.
        """
        if self.beeper_state() == 0:
            log.warning("Beeper is disabled, cannot generate beep sound.")
            return
        assert 55 <= frequency <= 6640, "Frequency must be between 55 and 6640 Hz"
        assert 0.05 <= duration <= 12.75, "Duration must be between 0.05 and 12.75 seconds"
        self.write(f"SYST:BEEP {frequency},{duration}")
