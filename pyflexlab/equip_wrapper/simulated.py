from typing import Literal
import threading
import numpy as np
import time
from . import ACSourceMeter, DCSourceMeter, ITC, Magnet
from pyomnix.omnix_logger import get_logger
from pyomnix.utils.math import SWITCH_DICT
from pyomnix.utils import CacheArray
from pyomnix.utils import print_progress_bar

logger = get_logger(__name__)


class SimMeter(ACSourceMeter, DCSourceMeter):
    def __init__(self, *args, **kwargs):
        self.meter = None
        self.info_dict = {
            "output_status": False,
            "ac_dc": "dc",
            "output": {
                "voltage": 120,
                "current": 0,
                "frequency": None,
            },
        }
        self.output_target = 0
        self.safe_step = {"curr": 2e-3, "volt": 1e-1}
        self._source_range = 1
        self.sense_range = {"curr": 1, "volt": 1}

    @property
    def sense_range_curr(self):
        return self.sense_range["curr"]

    @sense_range_curr.setter
    def sense_range_curr(self, value: float):
        self.sense_range["curr"] = value
        
    @property
    def sense_range_volt(self):
        return self.sense_range["volt"]

    @sense_range_volt.setter
    def sense_range_volt(self, value: float):
        self.sense_range["volt"] = value

    @property
    def source_range(self):
        return self._source_range

    @source_range.setter
    def source_range(self, value: float):
        self._source_range = value

    def info_sync(self):
        pass

    def setup(self, 
              ac_dc: Literal["ac", "dc"] | None = None, 
              *args, 
              reset: bool = False,
              **kwargs):
        if reset is True and ac_dc is None:
            ac_dc = "dc"
        if ac_dc is not None:
            self.info_dict["ac_dc"] = ac_dc

    def output_switch(self, switch: bool | Literal["on", "off", "ON", "OFF"]):
        switch = SWITCH_DICT.get(switch, False) if isinstance(switch, str) else switch
        if self.info_dict.get("output_status") == switch:
            logger.info(f"Output status is already {switch}")
            return
        if switch == "on" or switch == "ON" or switch == True:
            logger.info("start output")
            self.info_dict["output_status"] = True
        elif switch == "off" or switch == "OFF" or switch == False:
            logger.info("stop output")
            self.info_dict["output_status"] = False
        else:
            raise ValueError(f"Invalid switch value: {switch}")

    def get_output_status(self) -> tuple[float, float, float] | tuple[float, float]:
        return (self.output_target, self.output_target, self.output_target)

    def sense(self, type_str: Literal["curr"] | Literal["volt"]) -> float:
        if self.info_dict["ac_dc"] == "dc":
            return np.random.randn()
        else:
            x = np.random.randn()
            y = np.random.randn()
            return x, y, np.sqrt(x**2 + y**2), np.arctan(np.divide(y, x)) * 180 / np.pi

    def uni_output(
        self,
        value: float | str,
        *,
        freq: float | str | None = None,
        compliance: float | str | None = None,
        fix_range: float | str | None = None,
        type_str: Literal["curr"] | Literal["volt"],
        alter_range: bool = False,
    ) -> float:
        if freq is not None and self.info_dict["ac_dc"] == "dc":
            raise ValueError("freq should be None for dc output")
        elif freq is None and self.info_dict["ac_dc"] == "ac":
            raise ValueError("freq should not be None for ac output")
        if type_str == "curr":
            self.info_dict["output"]["current"] = value
            self.info_dict["output"]["frequency"] = freq
            self.output_target = value
        elif type_str == "volt":
            self.info_dict["output"]["voltage"] = value
            self.info_dict["output"]["frequency"] = freq
            self.output_target = value
        else:
            raise ValueError(f"Invalid type_str: {type_str}")
        return value

    def rms_output(
        self,
        value: float | str,
        *,
        freq: float | str | None = None,
        compliance: float | str | None = None,
        fix_range: float | str | None = None,
        type_str: Literal["curr"] | Literal["volt"],
    ) -> float:
        pass

    def dc_output(
        self,
        value: float | str,
        *,
        type_str: Literal["curr"] | Literal["volt"],
    ) -> float:
        pass

    def shutdown(self):
        pass


class SimMagnet(Magnet):
    def __init__(self, *args, **kwargs):
        pass


class SimITC(ITC):
    def __init__(self, *args, **kwargs):
        self.temp = 300
        self.cache = CacheArray(10)
        self.temp_set = 300

    def begin_vary(self, vary_spd: float = 3):
        """
        The temperature will gradually approach the set point
        to simulate real-world behavior.
        """

        def _vary_temp():
            t = 0
            temp_ini = self.temp
            while t < 100:
                self.temp = self.temp_set + (temp_ini - self.temp_set) * np.exp(
                    -t / vary_spd
                )
                time.sleep(1)
                t += 1
            self.temp = self.temp_set

        # Start temperature variation in a separate thread
        threading.Thread(target=_vary_temp, daemon=True).start()

    def get_temperature(self) -> float:
        return self.temp

    @property
    def temperature_set(self) -> float:
        return self.temp_set

    @temperature_set.setter
    def temperature_set(self, temp: float):
        self.temp_set = temp
        self.begin_vary()

    @property
    def pid(self):
        return self.pid

    def set_pid(self, pid_dict):
        self.pid[0] = pid_dict.get("p", 1)
        self.pid[1] = pid_dict.get("i", 1)
        self.pid[2] = pid_dict.get("d", 1)

    def correction_ramping(
        self,
        temp: float,
        trend: Literal["up"]
        | Literal["down"]
        | Literal["up-huge"]
        | Literal["down-huge"],
    ):
        pass


class SimMag(Magnet):
    def __init__(self, *args, **kwargs):
        self.f = 0
        self.f_set = 0
        self.cache = CacheArray(30)

    def begin_vary(self):
        """
        The temperature will gradually approach the set point
        to simulate real-world behavior.
        """

        def _vary_field():
            t = 0
            field_ini = self.f
            while t < 100:
                self.f = self.f_set + (field_ini - self.f_set) * np.exp(-t / 10)
                time.sleep(1)
                t += 1
            self.f = self.f_set

        # Start temperature variation in a separate thread
        threading.Thread(target=_vary_field, daemon=True).start()

    # make field an alias of temperature for convenience
    @property
    def field(self) -> float:
        self.cache.update_cache(self.f)
        return self.f

    @property
    def field_set(self) -> float:
        return self.f_set

    @field_set.setter
    def field_set(self, value: float) -> None:
        self.f_set = value

    @property
    def status(self) -> Literal["TO SET", "HOLD"]:
        """return the varying status of the ITC"""
        status_return = self.cache.get_status()
        if status_return is None:
            return "TO SET"
        return "HOLD" if status_return["if_stable"] else "TO SET"

    def if_reach_target(self, tolerance: float = 3e-3):
        """
        check if the magnet has reached the target field

        Args:
            tolerance (float): the tolerance of the field (T)
        """
        return abs(self.field - self.field_set) < tolerance

    def ramp_to_field(
        self,
        field: float | int | tuple[float] | list[float],
        *,
        rate: float | tuple[float] = (0.2,) * 3,
        wait: bool = True,
        tolerance: float = 1e-3,
    ) -> None:
        """
        ramp the magnetic field to the target value with the rate, current the field is only in Z direction limited by the actual instrument setting
        (currently only B_z can be ramped)

        Args:
            field (tuple[float]): the target field coor
            rate (float): the rate of the field change (T/min)
            wait (bool): whether to wait for the ramping to finish
            tolerance (float): the tolerance of the field (T)
        """
        if abs(self.field - field) < tolerance:
            return
        if isinstance(rate, (float, int)):
            assert rate <= 0.2, "The rate is too high, the maximum rate is 0.2 T/min"
        else:
            assert max(rate) <= 0.2, (
                "The rate is too high, the maximum rate is 0.2 T/min"
            )
        ini_field = self.field
        self.field_set = field
        self.begin_vary()

        if wait:
            while (
                self.status == "TO SET" or abs(self.field - self.field_set) > tolerance
            ):
                print_progress_bar(
                    self.field - ini_field,
                    field - ini_field,
                    prefix="Stablizing",
                    suffix=f"B: {self.field} T",
                )
                time.sleep(1)
            logger.info("ramping finished")


class FakeMag(Magnet):
    """
    a fake magnet that always gives zero field
    """

    def __init__(self, *args, **kwargs):
        self.cache = CacheArray(30)

    def ramp_to_field(
        self,
        field: float | int | tuple[float] | list[float],
        *,
        rate: float | tuple[float] = (0.2,) * 3,
        wait: bool = True,
        tolerance: float = 1e-3,
    ) -> None:
        if field != 0:
            logger.warning("You are ramping a fake magnet, the field will not change")

    @property
    def field(self) -> float:
        return 0

    @property
    def field_set(self) -> float:
        return 0

    @field_set.setter
    def field_set(self, value: float) -> None:
        pass

    def status(self) -> Literal["TO SET", "HOLD"]:
        return "HOLD"

    def if_reach_target(self, tolerance: float = 3e-3) -> bool:
        return True


FakeITC = SimITC
