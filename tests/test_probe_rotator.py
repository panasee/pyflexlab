import ctypes
from pathlib import Path

import pytest

from pyflexlab.drivers import probe_rotator


def int_value(value) -> int:
    return value.value if hasattr(value, "value") else int(value)


class FakeWJApi:
    def __init__(self) -> None:
        self.calls = []
        self.current_pulse = 12500
        self.pulse_sequence = []
        self.status = 0
        self.status_sequence = []
        self.speed = 2
        self.fail_get_pulse = False
        self.fail_get_pulse_times = 0
        self.fail_status = False
        self.fail_status_times = 0
        self.fail_get_vel = False
        self.fail_get_vel_times = 0
        self.fail_move = False
        for name in (
            "WJ_Open",
            "WJ_Close",
            "WJ_Get_Axis_Status",
            "WJ_Get_Axis_Pulses",
            "WJ_Get_Axes_Pulses",
            "WJ_Get_Axis_Vel",
            "WJ_Set_Axis_Vel",
            "WJ_Move_Axis_Pulses",
            "WJ_Move_Axis_Emergency_Stop",
            "WJ_Get_Axis_Acc",
            "WJ_Get_Axis_Dec",
            "WJ_Get_Axis_Subdivision",
            "WJ_Get_Axes_Status",
            "WJ_Get_Axes_Num",
            "WJ_Move_Axes_Pulses",
            "WJ_Move_Axis_Vel",
            "WJ_Move_Axes_Vel",
            "WJ_Move_Axis_Slow_Stop",
            "WJ_Move_Axis_Home",
            "WJ_Set_Axis_Acc",
            "WJ_Set_Axis_Dec",
            "WJ_Set_Axis_Vel",
            "WJ_Set_Axis_Subdivision",
            "WJ_Set_Axis_Slow_Stop",
            "WJ_Set_Led_Twinkle",
            "WJ_Set_Axis_Pulses_Zero",
            "WJ_Set_Default",
            "WJ_Set_Move_Axis_Vel_Acc",
            "WJ_Set_Axis_Home_Pulses",
            "WJ_IO_Output",
            "WJ_IO_Input",
        ):
            setattr(self, name, FakeFunction(getattr(self, f"_{name}")))

    def _WJ_Open(self, port):
        self.calls.append(("open", int_value(port)))
        return 0

    def _WJ_Close(self):
        self.calls.append(("close",))
        return 0

    def _WJ_Get_Axis_Status(self, axis, status_ptr):
        self.calls.append(("get_axis_status", int_value(axis)))
        if self.fail_status_times:
            self.fail_status_times -= 1
            return 1
        if self.fail_status:
            return 1
        status_ptr._obj.value = (
            self.status_sequence.pop(0) if self.status_sequence else self.status
        )
        return 0

    def _WJ_Get_Axis_Pulses(self, axis, pulse_ptr):
        self.calls.append(("get_axis_pulses", int_value(axis)))
        if self.fail_get_pulse_times:
            self.fail_get_pulse_times -= 1
            return 1
        if self.fail_get_pulse:
            return 1
        pulse_ptr._obj.value = (
            self.pulse_sequence.pop(0) if self.pulse_sequence else self.current_pulse
        )
        return 0

    def _WJ_Get_Axes_Pulses(self, pulse_array):
        self.calls.append(("get_axes_pulses", len(pulse_array)))
        pulse_array[0] = 0
        return 0

    def _WJ_Get_Axis_Vel(self, axis, speed_ptr):
        self.calls.append(("get_axis_vel", int_value(axis)))
        if self.fail_get_vel_times:
            self.fail_get_vel_times -= 1
            return 1
        if self.fail_get_vel:
            return 1
        speed_ptr._obj.value = self.speed
        return 0

    def _WJ_Set_Axis_Vel(self, axis, value):
        self.calls.append(("set_axis_vel", int_value(axis), int_value(value)))
        self.speed = int_value(value)
        return 0

    def _WJ_Move_Axis_Pulses(self, axis, delta):
        self.calls.append(("move_axis_pulses", int_value(axis), int_value(delta)))
        return 1 if self.fail_move else 0

    def _WJ_Move_Axis_Emergency_Stop(self, axis):
        self.calls.append(("emergency_stop", int_value(axis)))
        return 0

    def _default_success(self, *args):
        self.calls.append(("unused", args))
        return 0

    _WJ_Get_Axis_Acc = _default_success
    _WJ_Get_Axis_Dec = _default_success
    _WJ_Get_Axis_Subdivision = _default_success
    _WJ_Get_Axes_Status = _default_success
    _WJ_Get_Axes_Num = _default_success
    _WJ_Move_Axes_Pulses = _default_success
    _WJ_Move_Axis_Vel = _default_success
    _WJ_Move_Axes_Vel = _default_success
    _WJ_Move_Axis_Slow_Stop = _default_success
    _WJ_Move_Axis_Home = _default_success
    _WJ_Set_Axis_Acc = _default_success
    _WJ_Set_Axis_Dec = _default_success
    _WJ_Set_Axis_Subdivision = _default_success
    _WJ_Set_Axis_Slow_Stop = _default_success
    _WJ_Set_Led_Twinkle = _default_success
    _WJ_Set_Axis_Pulses_Zero = _default_success
    _WJ_Set_Default = _default_success
    _WJ_Set_Move_Axis_Vel_Acc = _default_success
    _WJ_Set_Axis_Home_Pulses = _default_success
    _WJ_IO_Output = _default_success
    _WJ_IO_Input = _default_success


class FakeFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


def make_rotator(monkeypatch):
    fake_api = FakeWJApi()
    monkeypatch.setattr(probe_rotator.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        probe_rotator.RotatorProbe,
        "_resolve_dll_path",
        staticmethod(lambda: Path(__file__)),
    )
    monkeypatch.setattr(ctypes, "WinDLL", lambda _path: fake_api)

    rotator = probe_rotator.RotatorProbe()

    return rotator, fake_api


def test_curr_angle_uses_single_axis_pulse_query(monkeypatch):
    rotator, fake_api = make_rotator(monkeypatch)

    assert rotator.curr_angle() == 90
    assert ("get_axis_pulses", 1) in fake_api.calls
    assert not any(call[0] == "get_axes_pulses" for call in fake_api.calls)


def test_curr_angle_raises_when_pulse_query_fails(monkeypatch):
    rotator, fake_api = make_rotator(monkeypatch)
    fake_api.fail_get_pulse = True

    with pytest.raises(RuntimeError, match="WJ_Get_Axis_Pulses failed"):
        rotator.curr_angle()


def test_curr_angle_reconnects_once_when_query_session_is_stale(monkeypatch):
    rotator, fake_api = make_rotator(monkeypatch)
    fake_api.fail_get_pulse_times = 1

    assert rotator.curr_angle() == 90
    assert fake_api.calls.count(("close",)) == 1
    assert fake_api.calls.count(("open", 0)) == 2


def test_curr_angle_error_mentions_stale_session_after_reconnect_fails(monkeypatch):
    rotator, fake_api = make_rotator(monkeypatch)
    fake_api.fail_get_pulse = True

    with pytest.raises(RuntimeError, match="session may be stale"):
        rotator.curr_angle()


def test_if_running_raises_when_status_query_fails(monkeypatch):
    rotator, fake_api = make_rotator(monkeypatch)
    fake_api.fail_status = True

    with pytest.raises(RuntimeError, match="WJ_Get_Axis_Status failed"):
        rotator.if_running()


def test_if_running_reconnects_once_when_status_session_is_stale(monkeypatch):
    rotator, fake_api = make_rotator(monkeypatch)
    fake_api.fail_status_times = 1

    assert rotator.if_running() is False
    assert fake_api.calls.count(("close",)) == 1
    assert fake_api.calls.count(("open", 0)) == 2


def test_is_stale_returns_false_without_reconnecting_for_ok_query(monkeypatch):
    rotator, fake_api = make_rotator(monkeypatch)

    assert rotator.is_stale() is False
    assert fake_api.calls.count(("close",)) == 0
    assert fake_api.calls.count(("open", 0)) == 1


def test_is_stale_returns_true_without_reconnecting_for_failed_query(monkeypatch):
    rotator, fake_api = make_rotator(monkeypatch)
    fake_api.fail_status = True

    assert rotator.is_stale() is True
    assert fake_api.calls.count(("close",)) == 0
    assert fake_api.calls.count(("open", 0)) == 1


def test_spd_reconnects_once_when_query_session_is_stale(monkeypatch):
    rotator, fake_api = make_rotator(monkeypatch)
    fake_api.fail_get_vel_times = 1

    assert rotator.spd() == 2
    assert fake_api.calls.count(("close",)) == 1
    assert fake_api.calls.count(("open", 0)) == 2


def test_ramp_angle_raises_when_move_fails_without_updating_target(monkeypatch):
    rotator, fake_api = make_rotator(monkeypatch)
    fake_api.fail_move = True

    with pytest.raises(RuntimeError, match="WJ_Move_Axis_Pulses failed"):
        rotator.ramp_angle(180, wait=False)

    assert rotator.angle_set is None


def test_ramp_angle_sets_target_after_move_is_accepted(monkeypatch):
    rotator, _fake_api = make_rotator(monkeypatch)

    rotator.ramp_angle(180, wait=False)

    assert rotator.angle_set == 180


def test_ramp_angle_waits_for_target_when_status_starts_false(monkeypatch):
    rotator, fake_api = make_rotator(monkeypatch)
    monkeypatch.setattr(probe_rotator.time, "sleep", lambda _seconds: None)
    fake_api.status_sequence = [0, 0, 0]
    fake_api.pulse_sequence = [12500, 12500, 12639]

    rotator.ramp_angle(91, wait=True)

    assert fake_api.calls.count(("get_axis_pulses", 1)) == 3
    assert rotator.angle_set == 91
