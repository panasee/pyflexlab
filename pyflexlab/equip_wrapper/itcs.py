import time
from typing import Literal, Optional
from typing_extensions import override

from abc import ABC, abstractmethod

from pymeasure.instruments.oxfordinstruments import ITC503
from qcodes.instrument_drivers.Lakeshore import LakeshoreModel336
from pyomnix.omnix_logger import get_logger
from pyomnix.utils import convert_unit, print_progress_bar, CacheArray

from ..drivers.mercuryITC import MercuryITC

logger = get_logger(__name__)


"""
Wrappers for ITC are following
"""


class ITC(ABC):
    # parent class to incorporate both two ITCs
    @abstractmethod
    def __init__(
        self,
        address: str,
        cache_length: int = 60,
        var_crit: float = 1e-4,
        least_length: int = 13,
    ):
        self.cache = CacheArray(
            cache_length=cache_length, var_crit=var_crit, least_length=least_length
        )

    def set_cache(self, *, cache_length: int, var_crit: Optional[float] = None):
        """
        set the cache for the ITC
        """
        if var_crit is None:
            self.cache = CacheArray(cache_length=cache_length)
        else:
            self.cache = CacheArray(cache_length=cache_length, var_crit=var_crit)

    @property
    def temperature(self) -> float:
        """return the precise temperature of the sample"""
        temp: float = self.get_temperature()
        self.cache.update_cache(temp)
        return temp

    def add_cache(self) -> None:
        """add the temperature to the cache without returning"""
        temp: float = self.get_temperature()
        self.cache.update_cache(temp)

    @abstractmethod
    def get_temperature(self) -> float:
        """get the temperature from the instrument without caching"""

    def load_cache(self, load_length: int = 30) -> None:
        """load the cache from the instrument"""
        for i in range(load_length):
            self.add_cache()
            print_progress_bar(i + 1, load_length, prefix="loading cache")
            time.sleep(1)

    @property
    def status(self) -> Literal["VARYING", "HOLD"]:
        """return the varying status of the ITC"""
        status_return = self.cache.get_status()
        return "HOLD" if status_return["if_stable"] else "VARYING"

    @property
    @abstractmethod
    def temperature_set(self):
        """return the setpoint temperature"""
        pass

    @temperature_set.setter
    @abstractmethod
    def temperature_set(self, temp):
        """
        set the target temperature for sample, as for other parts' temperature, use the methods for each ITC
        """
        pass

    @property
    @abstractmethod
    def pid(self):
        """
        return the PID parameters
        """
        pass

    @abstractmethod
    def set_pid(self, pid_dict):
        """
        set the PID parameters

        Args:
            pid_dict (Dict): a dictionary as {"P": float, "I": float, "D": float}
        """
        pass

    @abstractmethod
    def correction_ramping(
        self, temp: float, trend: Literal["up", "down", "up-huge", "down-huge"]
    ):
        """
        Correct the sensor choosing or pressure when ramping through the temperature threshold

        Args:
            temp (float): the current temperature
            trend (Literal["up","down"]): the trend of the temperature
        """
        pass

    def wait_for_temperature(
        self,
        temp,
        *,
        check_interval=1,
        stability_counter=1,
        thermalize_counter=7,
        correction_needed=False,
    ):
        """
        wait for the temperature to stablize for a certain time length

        Args:
            temp (float): the target temperature
            check_interval (int,[s]): the interval to check the temperature
            stability_counter (int): the number of times the temperature is within the delta range
                to consider the temperature stablized
            thermalize_counter (int): the number of times to thermalize the sample
        """

        def tolerance_T(T: float):
            """
            set the tolerance to judge if ramping is needed
            """
            if T > 10:
                return T / 1000
            else:
                return 0.007

        trend: Literal["up", "down", "up-huge", "down-huge"]
        initial_temp = self.temperature
        if abs(initial_temp - temp) < tolerance_T(temp):
            return
        elif initial_temp < temp - 100:
            trend = "up-huge"
        elif initial_temp > temp + 100:
            trend = "down-huge"
        elif initial_temp < temp:
            trend = "up"
        else:
            trend = "down"

        i = 0
        while i < stability_counter:
            # self.add_cache()
            if correction_needed:
                self.correction_ramping(self.temperature, trend)
            if (
                abs(self.cache.get_status()["mean"] - temp) < ITC.dynamic_delta(temp)
                and self.cache.get_status()["if_stable"]
            ):
                i += 1
            else:
                i = 0
            print_progress_bar(
                self.temperature - initial_temp,
                temp - initial_temp,
                prefix="Stablizing",
                suffix=f"Temperature: {self.temperature:.3f} K",
            )
            time.sleep(check_interval)
        logger.info("Temperature stablized")
        for i in range(thermalize_counter):
            print_progress_bar(
                i + 1,
                thermalize_counter,
                prefix="Thermalizing",
                suffix=f"Temperature: {self.temperature:.3f} K",
            )
            time.sleep(check_interval)
        logger.info("Thermalizing finished")

    def ramp_to_temperature(
        self,
        temp,
        *,
        delta=0.02,
        check_interval=1,
        stability_counter=1,
        thermalize_counter=7,
        pid: Optional[dict] = None,
        ramp_rate=None,
        wait=True,
    ):
        """ramp temperature to the target value (not necessary sample temperature)"""
        self.temperature_set = temp
        if pid is not None:
            self.set_pid(pid)
        if wait:
            self.wait_for_temperature(
                temp,
                check_interval=check_interval,
                stability_counter=stability_counter,
                thermalize_counter=thermalize_counter,
            )

    @staticmethod
    def dynamic_delta(temp) -> float:
        """
        calculate a dynamic delta to help high temperature to stabilize (reach 0.1K tolerance when 300K and {delta_lowt} when 10K)
        """
        # linear interpolation
        delta_hight = 0.3
        t_high = 300
        delta_lowt = 0.02
        t_low = 1.5
        return (delta_hight - delta_lowt) * (temp - t_low) / (
            t_high - t_low
        ) + delta_lowt


class ITCLakeshore(ITC):
    def __init__(
        self,
        address: str = "GPIB0::12::INSTR",
        cache_length: int = 60,
        var_crit: float = 5e-4,
        least_length: int = 13,
    ):
        self.ls = LakeshoreModel336("Lakeshore336", address)
        self.cache = CacheArray(
            cache_length=cache_length, var_crit=var_crit, least_length=least_length
        )
        self.channels_no = len(self.ls.channels)
        self.second_stage = self.ls.C
        self.sample_mount = self.ls.B
        self.sample = self.ls.A
        self.heater_intrinsic = [self.ls.output_1, self.ls.output_2]
        self.binding = {"heater_1": "sample", "heater_2": "second_stage"}
        self.binding_inv = {v: k for k, v in self.binding.items()}
        if "sample" not in self.binding_inv and "sample_mount" not in self.binding_inv:
            logger.raise_error(
                "sample related sensor is not in the binding", ValueError
            )
        self.heater_sample_str = (
            self.binding_inv["sample"]
            if "sample" in self.binding_inv
            else self.binding_inv["sample_mount"]
        )
        self.heater_sample = (
            self.ls.output_1
            if self.heater_sample_str == "heater_1"
            else self.ls.output_2
        )
        self._bind_heater()

    def get_binding(self):
        print(f"Heater 1 is bound to {self.binding['heater_1']}")
        print(f"Heater 2 is bound to {self.binding['heater_2']}")

    def change_binding(self, *, heater_1: str, heater_2: str):
        logger.warning("usually not needed, be aware of what you are doing")
        self.binding["heater_1"] = heater_1
        self.binding["heater_2"] = heater_2
        self._bind_heater()

    def _bind_heater(self):
        if self.binding["heater_1"] == self.binding["heater_2"]:
            logger.raise_error("Heater 1 and Heater 2 cannot be the same", ValueError)
        match self.binding["heater_1"]:
            case "sample":
                self.ls.output_1.input_channel("A")
            case "second_stage":
                self.ls.output_1.input_channel("C")
            case "sample_mount":
                self.ls.output_1.input_channel("B")
            case _:
                logger.raise_error(
                    f"Invalid heater binding: {self.binding['heater_1']}", ValueError
                )
        match self.binding["heater_2"]:
            case "sample":
                self.ls.output_2.input_channel("A")
            case "second_stage":
                self.ls.output_2.input_channel("C")
            case "sample_mount":
                self.ls.output_2.input_channel("B")
            case _:
                logger.raise_error(
                    f"Invalid heater binding: {self.binding['heater_2']}", ValueError
                )

    def get_temperature(self) -> float:
        return self.sample.temperature()

    @property
    def temperature_set(self) -> float:
        return self.heater_sample.setpoint()

    @temperature_set.setter
    def temperature_set(self, temp: float) -> None:
        self.heater_sample.setpoint(temp)

    @property
    def pid(self) -> dict:
        return {
            "P": self.heater_sample.P(),
            "I": self.heater_sample.I(),
            "D": self.heater_sample.D(),
        }

    def set_pid(self, pid_dict: dict) -> None:
        self.heater_sample.P(pid_dict["P"])
        self.heater_sample.I(pid_dict["I"])
        self.heater_sample.D(pid_dict["D"])

    def correction_ramping(
        self,
        temp: float | str,
        trend: Literal["up"]
        | Literal["down"]
        | Literal["up-huge"]
        | Literal["down-huge"],
    ):
        pass

    def ramp_to_temperature(
        self,
        temp: float | str,
        *,
        delta=0.02,
        check_interval=1,
        stability_counter=1,
        thermalize_counter=7,
        pid: dict | None = None,
        ramp_rate=None,
        wait=True,
    ):
        temp = convert_unit(temp, "K")[0]
        self.temperature_set = temp
        if pid is not None:
            self.set_pid(pid)
        if ramp_rate is not None:
            self.heater_sample.setpoint_ramp_rate(ramp_rate)

        self.heater_sample.output_range("medium")
        if wait:
            self.wait_for_temperature(
                temp,
                check_interval=check_interval,
                stability_counter=stability_counter,
                thermalize_counter=thermalize_counter,
            )

    @override
    @property
    def status(self) -> Literal["VARYING", "HOLD"]:
        """return the varying status of the ITC"""
        status_return = self.cache.get_status()
        if status_return["if_stable"] and not self.heater_sample.setpoint_ramp_status():
            return "HOLD"
        else:
            return "VARYING"


class ITCMercury(ITC):
    """
    Variable Params:
    self.correction_ramping: modify pressure according to the temperature and trend
    self.calculate_vti_temp (in driver): automatically calculate the set VTI temperature
    """

    def __init__(
        self,
        address="TCPIP0::10.97.27.13::7020::SOCKET",
        cache_length: int = 60,
        var_crit: float = 5e-4,
        least_length: int = 13,
    ):
        self.mercury = MercuryITC("mercury_itc", address)
        self.cache = CacheArray(
            cache_length=cache_length, var_crit=var_crit, least_length=least_length
        )

    @property
    def pres(self):
        return self.mercury.pressure()

    def set_pres(self, pres: float):
        self.mercury.pressure_setpoint(pres)

    @property
    def flow(self):
        return self.mercury.gas_flow()

    def set_flow(self, flow: float):
        """
        set the gas flow, note the input value is percentage, from 0 to 99.9 (%)
        """
        if not 0.0 < flow < 100.0:
            raise ValueError("Flow must be between 0.0 and 100.0 (%)")
        self.mercury.gas_flow(flow)

    @property
    def pid(self):
        return {
            "P": self.mercury.temp_loop_P(),
            "I": self.mercury.temp_loop_I(),
            "D": self.mercury.temp_loop_D(),
        }

    def set_pid(self, pid_dict: dict):
        """
        set the pid of probe temp loop
        """
        self.mercury.temp_PID = (pid_dict["P"], pid_dict["I"], pid_dict["D"])
        self.pid_control("ON")

    def pid_control(self, control: Literal["ON", "OFF"]):
        self.mercury.temp_PID_control(control)

    def get_temperature(self) -> float:
        return self.mercury.probe_temp()

    def set_temperature(self, temp, vti_diff=None):
        """set the target temperature for sample"""
        self.mercury.temp_setpoint(temp)
        if vti_diff is not None:
            self.mercury.vti_temp_setpoint(temp - vti_diff)
        else:
            self.mercury.vti_temp_setpoint(self.mercury.calculate_vti_temp(temp))

    @property
    def temperature_set(self):
        return self.mercury.temp_setpoint()

    @temperature_set.setter
    def temperature_set(self, temp):
        self.set_temperature(temp)

    @property
    def vti_temperature(self):
        return self.mercury.vti_temp()

    def set_vti_temperature(self, temp):
        self.mercury.vti_temp_setpoint(temp)

    def ramp_to_temperature(
        self,
        temp,
        *,
        check_interval=1,
        stability_counter=10,
        thermalize_counter=7,
        pid=None,
        ramp_rate=None,
        wait=True,
        vti_diff: Optional[float] = 5,
    ):
        """ramp temperature to the target value (not necessary sample temperature)

        Args:
            temp (float): the target temperature
            delta (float): the temperature difference to consider the temperature stablized
            check_interval (int,[s]): the interval to check the temperature
            stability_counter (int): the number of times the temperature is within the delta range to consider the temperature stablized
            thermalize_counter (int): the number of times to thermalize the sample
            pid (Dict): a dictionary as {"P": float, "I": float, "D": float}
            ramp_rate (float, [K/min]): the rate to ramp the temperature
            wait (bool): whether to wait for the ramping to finish
            vti_diff (float, None to ignore VTI): the difference between the sample temperature and the VTI temperature
        """
        temp = convert_unit(temp, "K")[0]
        self.temperature_set = temp
        if pid is not None:
            self.set_pid(pid)

        if ramp_rate is not None:
            self.mercury.probe_ramp_rate(ramp_rate)
            # self.mercury.vti_heater_rate(ramp_rate)
            self.mercury.probe_temp_ramp_mode(
                "ON"
            )  # ramp_mode means limited ramping rate mode
        else:
            self.mercury.probe_temp_ramp_mode("OFF")
        if wait:
            self.wait_for_temperature(
                temp,
                check_interval=check_interval,
                stability_counter=stability_counter,
                thermalize_counter=thermalize_counter,
            )

    def correction_ramping(
        self, temp: float, trend: Literal["up", "down", "up-huge", "down-huge"]
    ):
        """
        Correct the sensor choosing or pressure when ramping through the temperature threshold

        Args:
            temp (float): the current temperature
            trend (Literal["up","down","up-huge","down-huge"]): the trend of the temperature
        """
        if trend == "up-huge":
            self.set_flow(2)
        elif trend == "down-huge":
            if temp >= 5:
                self.set_flow(15)
            elif temp > 2:
                self.set_flow(8)
            else:
                self.set_flow(3)
        else:
            if temp <= 2.3:
                self.set_flow(2)
            if trend == "up":
                self.set_flow(3)
            else:
                self.set_flow(4)


class ITCs(ITC):
    """Represents the ITC503 Temperature Controllers and provides a high-level interface for interacting with the instruments.

    There are two ITC503 incorporated in the setup, named up and down. The up one measures the temperature of the heat switch(up R1), PT2(up R2), leaving R3 no specific meaning. The down one measures the temperature of the sorb(down R1), POT LOW(down R2), POT HIGH(down R3).
    """

    def __init__(
        self,
        address_up: str = "GPIB0::23::INSTR",
        address_down: str = "GPIB0::24::INSTR",
        clear_buffer=True,
        cache_length: int = 60,
        var_crit: float = 3e-4,
        least_length: int = 13,
    ):
        self.itc_up = ITC503(address_up, clear_buffer=clear_buffer)
        self.itc_down = ITC503(address_down, clear_buffer=clear_buffer)
        self.itc_up.control_mode = "RU"
        self.itc_down.control_mode = "RU"
        self.cache = CacheArray(
            cache_length=cache_length, var_crit=var_crit, least_length=least_length
        )

    def chg_display(self, itc_name, target):
        """
        This function is used to change the front display of the ITC503

        Parameters: itc_name (str): The name of the ITC503, "up" or "down" or "all" target (str):  'temperature
        setpoint', 'temperature 1', 'temperature 2', 'temperature 3', 'temperature error', 'heater',
        'heater voltage', 'gasflow', 'proportional band', 'integral action time', 'derivative action time',
        'channel 1 freq/4', 'channel 2 freq/4', 'channel 3 freq/4'.

        Returns:
        None
        """
        if itc_name == "all":
            self.itc_up.front_panel_display = target
            self.itc_down.front_panel_display = target
        elif itc_name == "up":
            self.itc_up.front_panel_display = target
        elif itc_name == "down":
            self.itc_down.front_panel_display = target

    @property
    def temperature_set(self):
        return self.itc_down.temperature_setpoint

    @temperature_set.setter
    def temperature_set(self, temp):
        """
        set the target temperature for sample, as for other parts' temperature, use the methods for each ITC

        Args:
            temp (float): the target temperature
            itc_name (Literal["up","down","all"]): the ITC503 to set the temperature
        """
        self.itc_down.temperature_setpoint = temp

    def ramp_to_temperature_selective(
        self, temp, itc_name: Literal["up", "down"], P=None, I=None, D=None
    ):
        """
        used to ramp the temperature of the ITCs, this method will wait for the temperature to stablize and thermalize for a certain time length
        """
        self.control_mode = ("RU", itc_name)
        if itc_name == "up":
            itc_here = self.itc_up
        elif itc_name == "down":
            itc_here = self.itc_down
        else:
            logger.error("Please specify the ITC to set")
            return
        itc_here.temperature_setpoint = temp
        if P is not None and I is not None and D is not None:
            itc_here.auto_pid = False
            itc_here.proportional_band = P
            itc_here.integral_action_time = I
            itc_here.derivative_action_time = D
        else:
            itc_here.auto_pid = True
        itc_here.heater_gas_mode = "AM"
        logger.info(f"temperature setted to {temp}")

    @property
    def version(self):
        """Returns the version of the ITC503."""
        return [self.itc_up.version, self.itc_down.version]

    @property
    def control_mode(self):
        """Returns the control mode of the ITC503."""
        return [self.itc_up.control_mode, self.itc_down.control_mode]

    @control_mode.setter
    def control_mode(
        self, mode: tuple[Literal["LU", "RU", "LL", "RL"], Literal["all", "up", "down"]]
    ):
        """Sets the control mode of the ITC503. A two-element list is required. The second elecment is "all" or "up"
        or "down" to specify which ITC503 to set."""
        if mode[1] == "all":
            self.itc_up.control_mode = mode[0]
            self.itc_down.control_mode = mode[0]
        elif mode[1] == "up":
            self.itc_up.control_mode = mode[0]
        elif mode[1] == "down":
            self.itc_down.control_mode = mode[0]

    @property
    def heater_gas_mode(self):
        """Returns the heater gas mode of the ITC503."""
        return [self.itc_up.heater_gas_mode, self.itc_down.heater_gas_mode]

    @heater_gas_mode.setter
    def heater_gas_mode(
        self,
        mode: tuple[
            Literal["MANUAL", "AM", "MA", "AUTO"], Literal["all", "up", "down"]
        ],
    ):
        """Sets the heater gas mode of the ITC503. A two-element list is required. The second elecment is "all" or
        "up" or "down" to specify which ITC503 to set."""
        if mode[1] == "all":
            self.itc_up.heater_gas_mode = mode[0]
            self.itc_down.heater_gas_mode = mode[0]
        elif mode[1] == "up":
            self.itc_up.heater_gas_mode = mode[0]
        elif mode[1] == "down":
            self.itc_down.heater_gas_mode = mode[0]

    @property
    def heater_power(self):
        """Returns the heater power of the ITC503."""
        return [self.itc_up.heater, self.itc_down.heater]

    @property
    def heater_voltage(self):
        """Returns the heater voltage of the ITC503."""
        return [self.itc_up.heater_voltage, self.itc_down.heater_voltage]

    @property
    def gas_flow(self):
        """Returns the gasflow of the ITC503."""
        return [self.itc_up.gasflow, self.itc_down.gasflow]

    @property
    def proportional_band(self):
        """Returns the proportional band of the ITC503."""
        return [self.itc_up.proportional_band, self.itc_down.proportional_band]

    @property
    def integral_action_time(self):
        """Returns the integral action time of the ITC503."""
        return [self.itc_up.integral_action_time, self.itc_down.integral_action_time]

    @property
    def derivative_action_time(self):
        """Returns the derivative action time of the ITC503."""
        return [
            self.itc_up.derivative_action_time,
            self.itc_down.derivative_action_time,
        ]

    @property
    def pid(self):
        """Returns the PID of the ITC503."""
        return tuple(
            zip(
                self.proportional_band,
                self.integral_action_time,
                self.derivative_action_time,
            )
        )

    def set_pid(self, pid: dict, mode: Literal["all", "up", "down"] = "down"):
        """Sets the PID of the ITC503. A three-element list is required. The second elecment is "all" or "up" or "down" to specify which ITC503 to set.
        The P,I,D here are the proportional band (K), integral action time (min), and derivative action time(min), respectively.
        """
        self.control_mode = ("RU", mode)
        if mode == "all":
            self.itc_up.proportional_band = pid["P"]
            self.itc_down.proportional_band = pid["P"]
            self.itc_up.integral_action_time = pid["I"]
            self.itc_down.integral_action_time = pid["I"]
            self.itc_up.derivative_action_time = pid["D"]
            self.itc_down.derivative_action_time = pid["D"]
        if mode == "up":
            self.itc_up.proportional_band = pid["P"]
            self.itc_up.integral_action_time = pid["I"]
            self.itc_up.derivative_action_time = pid["D"]
        if mode == "down":
            self.itc_down.proportional_band = pid["P"]
            self.itc_down.integral_action_time = pid["I"]
            self.itc_down.derivative_action_time = pid["D"]

        if self.itc_up.proportional_band == 0:
            return ""
        return f"{mode} PID(power percentage): 100*(E/{pid['P']}+E/{pid['P']}*t/60{pid['I']}-dE*60{pid['D']}/{pid['P']}), [K,min,min]"

    @property
    def auto_pid(self):
        """Returns the auto pid of the ITC503."""
        return [self.itc_up.auto_pid, self.itc_down.auto_pid]

    @auto_pid.setter
    def auto_pid(self, mode):
        """Sets the auto pid of the ITC503. A two-element list is required. The second elecment is "all" or "up" or
        "down" to specify which ITC503 to set."""
        if mode[1] == "all":
            self.itc_up.auto_pid = mode[0]
            self.itc_down.auto_pid = mode[0]
        elif mode[1] == "up":
            self.itc_up.auto_pid = mode[0]
        elif mode[1] == "down":
            self.itc_down.auto_pid = mode[0]

    @property
    def temperature_setpoint(self):
        """Returns the temperature setpoint of the ITC503."""
        return [self.itc_up.temperature_setpoint, self.itc_down.temperature_setpoint]

    @temperature_setpoint.setter
    def temperature_setpoint(self, temperature):
        """Sets the temperature setpoint of the ITC503. A two-element list is required. The second elecment is "all"
        or "up" or "down" to specify which ITC503 to set."""
        if temperature[1] == "all":
            self.itc_up.temperature_setpoint = temperature[0]
            self.itc_down.temperature_setpoint = temperature[0]
        elif temperature[1] == "up":
            self.itc_up.temperature_setpoint = temperature[0]
        elif temperature[1] == "down":
            self.itc_down.temperature_setpoint = temperature[0]

    @property
    def temperatures(self):
        """Returns the temperatures of the whole device as a dict."""
        return {
            "sw": self.itc_up.temperature_1,
            "pt2": self.itc_up.temperature_2,
            "sorb": self.itc_down.temperature_1,
            "pot_low": self.itc_down.temperature_2,
            "pot_high": self.itc_down.temperature_3,
        }

    def get_temperature(self):
        """Returns the precise temperature of the sample"""
        if self.temperatures["pot_high"] < 1.9:
            return self.temperatures["pot_low"]
        elif self.temperatures["pot_high"] >= 1.9:
            return self.temperatures["pot_high"]

    def correction_ramping(
        self, temp: float, trend: Literal["up", "down", "up-huge", "down-huge"]
    ):
        pass
