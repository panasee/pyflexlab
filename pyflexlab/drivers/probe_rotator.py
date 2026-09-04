# this file requires the API WJ_API.dll to work
# by default it will look for the dll in the local DB directory
# for detailed information on the API see the WJ_API.h declarations
# make sure to use same architecture for the dll and python(32/64 bit here)
import functools
import platform
import time
from typing import Callable, Optional

from pathlib import Path
import ctypes
from .. import constants
from pyomnix.utils import print_progress_bar


def avoid_running(method):
    """
    Decorator to avoid running the function if the rotator is already running
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.if_running():
            print("Rotator is already running")
            return
        return method(self, *args, **kwargs)

    return wrapper


class RotatorProbe:
    # currently only one axis is used, so all methods are for one axis (axis_num: 1)
    def __init__(self, *, port: Optional[int] = None):
        assert platform.system().lower() == "windows", (
            "This module only works on Windows"
        )
        self.dll_path = self._resolve_dll_path()
        self._max_axes = 4  # LabVIEW reference flow passes a four-axis buffer.
        self.axis_num = 1
        self._upper_limit = 365
        self._lower_limit = -5
        self._to_zero_spd = -15
        # Controller velocity unit, measured on the current rotator as
        # roughly speed=1 -> 100 deg/min and speed=2 -> 200 deg/min.
        self.speed = 2
        self.angle_set: Optional[float] = None
        self._pulse_ratio = 50000  # 360 degrees / pulses
        self.serial_port = 0  # default serial port for usb
        if not self.dll_path.exists():
            raise FileNotFoundError(f"WJ_API.dll not found at {self.dll_path}")
        self.wj_api = ctypes.WinDLL(
            str(self.dll_path)
        )  # can pass Path-like after Python 3.12
        # define return type and arguments types for functions
        self.__declare_functions()
        if port is not None:
            self.status = self.connect(port)
        else:
            self.status = self.connect()

    def __del__(self):
        if hasattr(self, "wj_api"):
            self.exit()

    @staticmethod
    def _resolve_dll_path() -> Path:
        if constants.LOCAL_DB_PATH is None:
            return Path(".")
        return Path(constants.LOCAL_DB_PATH / "WJ_API.dll")

    @staticmethod
    def _check_status(status: int, function_name: str) -> None:
        if status != 0:
            raise RuntimeError(f"{function_name} failed with status {status}")

    def _query_with_reconnect(self, function_name: str, query: Callable[[], int]) -> None:
        status = query()
        if status == 0:
            return

        try:
            self.reconnect()
        except RuntimeError as exc:
            raise RuntimeError(
                f"{function_name} failed with status {status}; reconnect failed. "
                "Rotator session may be stale or another program may hold the device. "
                "Close LabVIEW/other clients and call reconnect()."
            ) from exc

        retry_status = query()
        if retry_status != 0:
            raise RuntimeError(
                f"{function_name} failed with status {retry_status} after reconnect. "
                "Rotator session may be stale or another program may hold the device. "
                "Close LabVIEW/other clients and call reconnect()."
            )

    def print_info(self):
        """print all info about the rotator"""
        print(f"Rotator is connected to serial port: {self.serial_port}")
        print(f"Rotator is running: {self.if_running()}")
        print(f"Rotator current angle: {self.curr_angle()}")
        print(f"Rotator current speed: {self.spd()}")

    def connect(self, serial_port: Optional[int] = None):
        """
        Connects to the rotator
        """
        if serial_port is not None:
            self.serial_port = serial_port
        print("Connecting Status:", status := self.wj_api.WJ_Open(self.serial_port))
        self._check_status(status, "WJ_Open")
        return status

    def reconnect(self):
        """
        Reopens the DLL/device session after a stale query connection.
        """
        self.wj_api.WJ_Close()
        return self.connect(self.serial_port)

    def is_stale(self, *, axis_no: Optional[int] = None) -> bool:
        """
        Debug helper that checks the DLL/device session without reconnecting.
        """
        if axis_no is None:
            axis_no = self.axis_num
        status = ctypes.c_int32()
        ret = self.wj_api.WJ_Get_Axis_Status(
            ctypes.c_int32(axis_no), ctypes.byref(status)
        )
        return ret != 0

    def exit(self):
        """
        Disconnects from the rotator
        """
        self.wj_api.WJ_Close()

    def if_running(self, *, axis_no: Optional[int] = None) -> bool:
        """
        Returns if the rotator is running
        """
        if axis_no is None:
            axis_no = self.axis_num
        status = ctypes.c_int32()

        self._query_with_reconnect(
            "WJ_Get_Axis_Status",
            lambda: self.wj_api.WJ_Get_Axis_Status(
                ctypes.c_int32(axis_no), ctypes.byref(status)
            ),
        )
        return status.value == 1

    def curr_angle(self, *, axis_no: Optional[int] = None) -> float:
        """
        Returns the current angle of the rotator
        """
        if axis_no is None:
            axis_no = self.axis_num
        pulse = ctypes.c_int32()

        self._query_with_reconnect(
            "WJ_Get_Axis_Pulses",
            lambda: self.wj_api.WJ_Get_Axis_Pulses(
                ctypes.c_int32(axis_no), ctypes.byref(pulse)
            ),
        )
        angle = pulse.value / self._pulse_ratio * 360
        # embed angle overflow control here
        if not (self._lower_limit <= angle <= self._upper_limit):
            self.emergency_stop()
            print(f"Rotator is at {angle}, reached its limit, emergency stop triggered")
        return angle

    def spd(self, *, axis_no: Optional[int] = None) -> int:
        """
        Returns the current speed of the rotator
        """
        if axis_no is None:
            axis_no = self.axis_num
        speed = ctypes.c_int32()

        self._query_with_reconnect(
            "WJ_Get_Axis_Vel",
            lambda: self.wj_api.WJ_Get_Axis_Vel(
                ctypes.c_int32(axis_no), ctypes.byref(speed)
            ),
        )
        self.speed = speed.value
        return speed.value

    @avoid_running
    def set_spd(self, value, *, axis_no: Optional[int] = None):
        """
        Sets the controller velocity unit, not an exact deg/s value.

        Empirical calibration on the current rotator:
        value=1 is about 100 deg/min; value=2 is about 200 deg/min.
        """
        if axis_no is None:
            axis_no = self.axis_num
        ret = self.wj_api.WJ_Set_Axis_Vel(
            ctypes.c_int32(axis_no), ctypes.c_int32(value)
        )
        self._check_status(ret, "WJ_Set_Axis_Vel")
        self.speed = value
        print("Speed set to: ", value)

    @avoid_running
    def ramp_angle(
        self,
        angle,
        *,
        progress=False,
        axis_no=None,
        wait=True,
        angle_tolerance=0.02,
        wait_timeout=None,
    ) -> None:
        """
        Moves the rotator to the specified angle

        Args:
            angle (in degrees, 360): the angle to move to
            axis_no: the axis number (1)
            wait: whether to wait for the motion to finish
            progress: (overwrite wait if True)whether to continuously monitor the motion
            angle_tolerance: finish tolerance in degrees for wait/progress
            wait_timeout: maximum wait time in seconds, or None to wait indefinitely
        """
        if axis_no is None:
            axis_no = self.axis_num
        initial_angle = self.curr_angle(axis_no=axis_no)
        delta_angle = angle - initial_angle
        delta_pulse = int(delta_angle * self._pulse_ratio / 360)
        ret = self.wj_api.WJ_Move_Axis_Pulses(
            ctypes.c_int32(axis_no), ctypes.c_int32(delta_pulse)
        )
        self._check_status(ret, "WJ_Move_Axis_Pulses")
        self.angle_set = angle
        if wait or progress:
            wait_started = time.monotonic()
            while True:
                running = self.if_running(axis_no=axis_no)
                current_angle = self.curr_angle(axis_no=axis_no)
                if progress:
                    print_progress_bar(
                        current_angle - initial_angle,
                        angle - initial_angle,
                    )
                if not running and abs(current_angle - angle) <= angle_tolerance:
                    break
                if (
                    wait_timeout is not None
                    and time.monotonic() - wait_started >= wait_timeout
                ):
                    raise TimeoutError(
                        f"Rotator did not reach {angle} deg within "
                        f"{wait_timeout} s; current angle is {current_angle} deg "
                        f"and running={running}"
                    )
                time.sleep(1)

    def emergency_stop(self, axis_no: int = 1):
        """
        Stops the rotator immediately
        """
        ret = self.wj_api.WJ_Move_Axis_Emergency_Stop(ctypes.c_int32(axis_no))
        self._check_status(ret, "WJ_Move_Axis_Emergency_Stop")

    def __declare_functions(self):
        """
        Declares used functions from the API
        """
        self.wj_api.WJ_Open.argtypes = [ctypes.c_int32]
        self.wj_api.WJ_Open.restype = ctypes.c_int32

        self.wj_api.WJ_Close.argtypes = []
        self.wj_api.WJ_Close.restype = ctypes.c_int32

        # Query Commands
        self.wj_api.WJ_Get_Axis_Acc.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.wj_api.WJ_Get_Axis_Acc.restype = ctypes.c_int32

        self.wj_api.WJ_Get_Axis_Dec.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.wj_api.WJ_Get_Axis_Dec.restype = ctypes.c_int32

        self.wj_api.WJ_Get_Axis_Vel.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.wj_api.WJ_Get_Axis_Vel.restype = ctypes.c_int32

        self.wj_api.WJ_Get_Axis_Subdivision.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.wj_api.WJ_Get_Axis_Subdivision.restype = ctypes.c_int32

        self.wj_api.WJ_Get_Axis_Status.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.wj_api.WJ_Get_Axis_Status.restype = ctypes.c_int32

        self.wj_api.WJ_Get_Axes_Status.argtypes = [
            ctypes.POINTER(ctypes.c_int32 * self._max_axes)
        ]
        self.wj_api.WJ_Get_Axes_Status.restype = ctypes.c_int32

        self.wj_api.WJ_Get_Axis_Pulses.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.wj_api.WJ_Get_Axis_Pulses.restype = ctypes.c_int32

        self.wj_api.WJ_Get_Axes_Pulses.argtypes = [
            ctypes.POINTER(ctypes.c_int32 * self._max_axes)
        ]
        self.wj_api.WJ_Get_Axes_Pulses.restype = ctypes.c_int32

        self.wj_api.WJ_Get_Axes_Num.argtypes = [ctypes.POINTER(ctypes.c_int32)]
        self.wj_api.WJ_Get_Axes_Num.restype = ctypes.c_int32

        # Motion Commands
        self.wj_api.WJ_Move_Axis_Pulses.argtypes = [ctypes.c_int32, ctypes.c_int32]
        self.wj_api.WJ_Move_Axis_Pulses.restype = ctypes.c_int32

        self.wj_api.WJ_Move_Axes_Pulses.argtypes = [
            ctypes.POINTER(ctypes.c_int32 * self._max_axes)
        ]
        self.wj_api.WJ_Move_Axes_Pulses.restype = ctypes.c_int32

        self.wj_api.WJ_Move_Axis_Vel.argtypes = [ctypes.c_int32, ctypes.c_int32]
        self.wj_api.WJ_Move_Axis_Vel.restype = ctypes.c_int32

        self.wj_api.WJ_Move_Axes_Vel.argtypes = [
            ctypes.POINTER(ctypes.c_int32 * self._max_axes)
        ]
        self.wj_api.WJ_Move_Axes_Vel.restype = ctypes.c_int32

        self.wj_api.WJ_Move_Axis_Emergency_Stop.argtypes = [ctypes.c_int32]
        self.wj_api.WJ_Move_Axis_Emergency_Stop.restype = ctypes.c_int32

        self.wj_api.WJ_Move_Axis_Slow_Stop.argtypes = [ctypes.c_int32]
        self.wj_api.WJ_Move_Axis_Slow_Stop.restype = ctypes.c_int32

        self.wj_api.WJ_Move_Axis_Home.argtypes = [ctypes.c_int32, ctypes.c_int32]
        self.wj_api.WJ_Move_Axis_Home.restype = ctypes.c_int32

        # Setting Commands
        self.wj_api.WJ_Set_Axis_Acc.argtypes = [ctypes.c_int32, ctypes.c_int32]
        self.wj_api.WJ_Set_Axis_Acc.restype = ctypes.c_int32

        self.wj_api.WJ_Set_Axis_Dec.argtypes = [ctypes.c_int32, ctypes.c_int32]
        self.wj_api.WJ_Set_Axis_Dec.restype = ctypes.c_int32

        self.wj_api.WJ_Set_Axis_Vel.argtypes = [ctypes.c_int32, ctypes.c_int32]
        self.wj_api.WJ_Set_Axis_Vel.restype = ctypes.c_int32

        self.wj_api.WJ_Set_Axis_Subdivision.argtypes = [ctypes.c_int32, ctypes.c_int32]
        self.wj_api.WJ_Set_Axis_Subdivision.restype = ctypes.c_int32

        self.wj_api.WJ_Set_Axis_Slow_Stop.argtypes = [ctypes.c_int32, ctypes.c_int32]
        self.wj_api.WJ_Set_Axis_Slow_Stop.restype = ctypes.c_int32

        self.wj_api.WJ_Set_Led_Twinkle.argtypes = []
        self.wj_api.WJ_Set_Led_Twinkle.restype = ctypes.c_int32

        self.wj_api.WJ_Set_Axis_Pulses_Zero.argtypes = [ctypes.c_int32]
        self.wj_api.WJ_Set_Axis_Pulses_Zero.restype = ctypes.c_int32

        self.wj_api.WJ_Set_Default.argtypes = []
        self.wj_api.WJ_Set_Default.restype = ctypes.c_int32

        self.wj_api.WJ_Set_Move_Axis_Vel_Acc.argtypes = [ctypes.c_int32, ctypes.c_int32]
        self.wj_api.WJ_Set_Move_Axis_Vel_Acc.restype = ctypes.c_int32

        self.wj_api.WJ_Set_Axis_Home_Pulses.argtypes = [ctypes.c_int32, ctypes.c_int32]
        self.wj_api.WJ_Set_Axis_Home_Pulses.restype = ctypes.c_int32

        # IO Commands
        self.wj_api.WJ_IO_Output.argtypes = [ctypes.c_int32, ctypes.c_int32]
        self.wj_api.WJ_IO_Output.restype = ctypes.c_int32

        self.wj_api.WJ_IO_Input.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.wj_api.WJ_IO_Input.restype = ctypes.c_int32
