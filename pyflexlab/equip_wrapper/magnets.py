"""
Wrappers for magnets are following
"""

import time
from typing import Literal, Optional

from abc import ABC, abstractmethod

import numpy as np
from pyomnix.omnix_logger import get_logger
from pyomnix.utils import print_progress_bar, SWITCH_DICT

from ..drivers.MercuryiPS_VISA import OxfordMercuryiPS

logger = get_logger(__name__)


class Magnet(ABC):
    @abstractmethod
    def __init__(self, address: str):
        pass

    @property
    @abstractmethod
    def field(self) -> float | tuple[float]:
        pass

    @property
    @abstractmethod
    def field_set(self):
        pass

    @field_set.setter
    @abstractmethod
    def field_set(self, field: float | tuple[float]):
        pass

    @abstractmethod
    def ramp_to_field(
        self, field: float, *, rate: float, stability: float, check_interval: float
    ):
        pass

    def if_reach_target(self, tolerance: float = 3e-3):
        """
        check if the magnet has reached the target field

        Args:
            tolerance (float): the tolerance of the field (T)
        """
        return abs(self.field - self.field_set) < tolerance


class WrapperIPS(Magnet):
    """
    Wrapper for MercuryIPS (only z axis magnetic field is considered)
    """

    def __init__(
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
        self.ips = OxfordMercuryiPS("mips", address)
        if if_print:
            self.ips.print_readable_snapshot(update=True)

        def spherical_limit(x, y, z) -> bool:
            return np.sqrt(x**2 + y**2 + z**2) <= limit_sphere

        self.ips.set_new_field_limits(spherical_limit)

    @property
    def field(self) -> float | tuple[float]:
        """
        return the current field of the magnet (only z direction considered)
        """
        return self.ips.z_measured()

    @property
    def field_set(self) -> float | tuple[float]:
        """
        set the target field (only z direction considered)
        """
        return self.ips.z_target()

    @field_set.setter
    def field_set(self, field: float | tuple[float]) -> None:
        """
        set the target field (only z direction considered)
        """
        assert isinstance(field, (float, int, tuple, list)), (
            "The field should be a float or a tuple of 3 floats"
        )
        fieldz_target = field if isinstance(field, (float, int)) else field[2]
        self.ips.z_target(fieldz_target)

    def sw_heater(
        self, switch: Optional[bool | Literal["on", "off", "ON", "OFF"]] = None
    ) -> Optional[bool]:
        """
        switch the heater of the magnet
        """
        if switch is not None:
            switch = (
                SWITCH_DICT.get(switch, False) if isinstance(switch, str) else switch
            )
            if switch:
                self.ips.GRPZ.sw_heater("ON")
            else:
                self.ips.GRPZ.sw_heater("OFF")
            logger.info("Heater switched %s", "on" if switch else "off")
        else:
            match self.ips.GRPZ.sw_heater():
                case "ON" | "on" | True:
                    return True
                case "OFF" | "off" | False:
                    return False
                case _:
                    raise ValueError("The heater status is not recognized")

    # =======suitable for Z-axis only ips========
    @property
    def status(self) -> Literal["HOLD", "TO SET", "CLAMP", "TO ZERO"]:
        """
        return the status of the magnet
        """
        return self.ips.GRPZ.ramp_status()

    @status.setter
    def status(self, status: Literal["HOLD", "TO SET", "CLAMP", "TO ZERO"]) -> None:
        """
        set the status of the magnet
        """
        assert status in ["HOLD", "TO SET", "CLAMP", "TO ZERO"], (
            "The status is not recognized"
        )
        self.ips.GRPZ.ramp_status(status)

    def ramp_to_field(
        self,
        field: float | int | tuple[float] | list[float],
        *,
        rate: float | tuple[float] = (0.2,) * 3,
        wait: bool = True,
        tolerance: float = 5e-3,
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
        if not self.sw_heater() and field != 0:
            self.sw_heater("on")
            for i in range(310):
                print_progress_bar(i, 310, prefix="waiting for heater")
                time.sleep(1)
        else:
            pass

        if abs(self.field - field) < tolerance:
            return
        if isinstance(rate, (float, int)):
            assert rate <= 0.2, "The rate is too high, the maximum rate is 0.2 T/min"
            self.ips.GRPZ.field_ramp_rate(rate / 60)
        else:
            assert max(rate) <= 0.2, (
                "The rate is too high, the maximum rate is 0.2 T/min"
            )
            self.ips.GRPZ.field_ramp_rate(rate[2] / 60)
        # self.ips.GRPX.field_ramp_rate(rate[0]/60)
        # self.ips.GRPY.field_ramp_rate(rate[1]/60)
        # no x and y field for now (see the setter method for details)
        ini_field = self.field
        self.field_set = field

        self.ips.ramp(mode="simul")
        if wait:
            # the is_ramping() method is not working properly, so we use the following method to wait for the ramping
            # to finish
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
