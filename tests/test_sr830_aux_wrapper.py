import pytest

import pyflexlab.equip_wrapper.meters as meters
from pyflexlab.equip_wrapper.meters import SourceMeter, WrapperSR830, WrapperSR830Aux
from pyflexlab.measure_manager import MeasureManager


class FakeSR830:
    def __init__(self):
        self.aux_out = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        self.aux_in = {1: 0.0, 2: -0.5, 3: 0.0, 4: 0.0}
        self.writes = []
        self.asks = []

    def write(self, command):
        self.writes.append(command)
        prefix, payload = command.split(" ", 1)
        assert prefix == "AUXV"
        channel, voltage = payload.split(",", 1)
        self.aux_out[int(channel.strip())] = float(voltage.strip())

    def ask(self, command):
        self.asks.append(command)
        prefix, channel = command.split()
        channel = int(channel)
        if prefix == "AUXV?":
            return str(self.aux_out[channel])
        if prefix == "OAUX?":
            return str(self.aux_in[channel])
        raise AssertionError(f"unexpected command: {command}")


class FakeSR830Wrapper:
    def __init__(self):
        self.meter = FakeSR830()


def test_sr830_aux_behaves_as_dc_voltage_source_meter():
    parent = FakeSR830Wrapper()
    aux = WrapperSR830Aux(parent, out_channel=1, in_channel=2)

    assert isinstance(aux, SourceMeter)

    aux.setup(function="source", source_type="volt")
    assert aux.uni_output(1.25, type_str="volt") == 1.25
    assert parent.meter.writes[-1] == "AUXV 1, 1.25"
    assert aux.get_output_status() == (1.25, 1.25)

    aux.setup(function="sense", sense_type="volt")
    assert aux.sense("volt") == -0.5

    aux.output_switch("off")
    assert parent.meter.aux_out[1] == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"type_str": "curr"},
        {"type_str": "volt", "freq": 17},
        {"type_str": "volt", "compliance": 1e-3},
    ],
)
def test_sr830_aux_rejects_unsupported_source_modes(kwargs):
    aux = WrapperSR830Aux(FakeSR830Wrapper(), out_channel=1, in_channel=1)

    with pytest.raises(ValueError):
        aux.uni_output(0.1, **kwargs)


def test_sr830_aux_rejects_invalid_channels_and_voltage_range():
    with pytest.raises(ValueError):
        WrapperSR830Aux(FakeSR830Wrapper(), out_channel=0, in_channel=1)

    aux = WrapperSR830Aux(FakeSR830Wrapper(), out_channel=1, in_channel=1)
    with pytest.raises(ValueError):
        aux.uni_output(10.6, type_str="volt")


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda: WrapperSR830Aux(FakeSR830Wrapper(), out_channel=0, in_channel=1),
            "out_channel must be one of 1, 2, 3, 4",
        ),
        (
            lambda: WrapperSR830Aux(FakeSR830Wrapper(), out_channel=1, in_channel=0),
            "in_channel must be one of 1, 2, 3, 4",
        ),
        (
            lambda: WrapperSR830Aux(FakeSR830Wrapper(), out_channel=1, in_channel=1).uni_output(10.6),
            "SR830 AUX OUT voltage must be between -10.5 V and 10.5 V",
        ),
        (
            lambda: WrapperSR830Aux(FakeSR830Wrapper(), out_channel=1, in_channel=1).setup(
                function="source", source_type="curr"
            ),
            "SR830 AUX OUT can only source voltage",
        ),
        (
            lambda: WrapperSR830Aux(FakeSR830Wrapper(), out_channel=1, in_channel=1).setup(
                function="sense", sense_type="curr"
            ),
            "SR830 AUX IN can only sense voltage",
        ),
        (
            lambda: WrapperSR830Aux(FakeSR830Wrapper(), out_channel=1, in_channel=1).setup(
                function="bad"
            ),
            "function should be either source or sense",
        ),
        (
            lambda: setattr(
                WrapperSR830Aux(FakeSR830Wrapper(), out_channel=1, in_channel=1),
                "sense_range_volt",
                1,
            ),
            "SR830 AUX IN voltage range is fixed",
        ),
        (
            lambda: WrapperSR830Aux(FakeSR830Wrapper(), out_channel=1, in_channel=1).sense_range_curr,
            "SR830 AUX IN cannot sense current",
        ),
        (
            lambda: setattr(
                WrapperSR830Aux(FakeSR830Wrapper(), out_channel=1, in_channel=1),
                "sense_range_curr",
                1,
            ),
            "SR830 AUX IN cannot sense current",
        ),
        (
            lambda: WrapperSR830Aux(FakeSR830Wrapper(), out_channel=1, in_channel=1).sense("curr"),
            "SR830 AUX IN can only sense voltage",
        ),
        (
            lambda: WrapperSR830Aux(FakeSR830Wrapper(), out_channel=1, in_channel=1).uni_output(
                0.1, type_str="curr"
            ),
            "SR830 AUX OUT can only source voltage",
        ),
        (
            lambda: WrapperSR830Aux(FakeSR830Wrapper(), out_channel=1, in_channel=1).uni_output(
                0.1, freq=17
            ),
            "SR830 AUX OUT is DC only and does not support frequency",
        ),
        (
            lambda: WrapperSR830Aux(FakeSR830Wrapper(), out_channel=1, in_channel=1).uni_output(
                0.1, compliance=1e-3
            ),
            "SR830 AUX OUT does not support compliance",
        ),
        (
            lambda: WrapperSR830Aux(FakeSR830Wrapper(), out_channel=1, in_channel=1).uni_output(
                0.1, fix_range=1
            ),
            "SR830 AUX OUT range is fixed",
        ),
    ],
)
def test_sr830_aux_uses_logger_raise_error_for_invalid_operations(
    monkeypatch, operation, message
):
    calls = []

    def fake_raise_error(message_text, exception_type, *args, **kwargs):
        calls.append((message_text, exception_type))
        raise exception_type(message_text)

    monkeypatch.setattr(meters.logger, "raise_error", fake_raise_error)

    with pytest.raises(ValueError, match=message):
        operation()

    assert calls == [(message, ValueError)]


def test_source_sweep_apply_accepts_sr830_aux_without_compliance():
    aux = WrapperSR830Aux(FakeSR830Wrapper(), out_channel=1, in_channel=1)
    manager = MeasureManager.__new__(MeasureManager)

    gen = manager.source_sweep_apply(
        "V",
        "dc",
        aux,
        max_value=0.1,
        step_value=0.1,
        compliance=None,
        sweepmode="0-max-0",
        source_wait=0,
    )

    assert next(gen) == 0
    assert next(gen) == 0.1
    assert aux.get_output_status() == (0.1, 0.1)


def test_sr830_wrapper_builds_associated_aux_meters():
    parent = WrapperSR830.__new__(WrapperSR830)
    parent.meter = FakeSR830()
    parent.info_dict = {"GPIB": "GPIB0::8::INSTR"}

    parent._init_aux_wrappers()

    assert sorted(parent.aux.keys()) == [1, 2, 3, 4]
    assert all(isinstance(aux, WrapperSR830Aux) for aux in parent.aux.values())
    assert parent.aux[1].meter is parent.meter

    parent.aux[4].uni_output(0.2, type_str="volt")
    assert parent.meter.aux_out[4] == 0.2


class FakeLoadedSR830:
    def __init__(self, address):
        self.address = address
        self.aux = {i: f"{address}-aux-{i}" for i in range(1, 5)}
        self.setup_calls = []

    def setup(self, **kwargs):
        self.setup_calls.append(kwargs)


def test_load_meter_keeps_sr830_aux_wrappers_on_parent_sr830():
    manager = MeasureManager.__new__(MeasureManager)
    manager.instrs = {}
    manager.meter_wrapper_dict = {"sr830": FakeLoadedSR830}

    manager.load_meter("sr830", "GPIB0::8::INSTR", "GPIB0::9::INSTR")

    assert len(manager.instrs["sr830"]) == 2
    assert "sr830_aux" not in manager.instrs
    assert manager.instrs["sr830"][0].aux[1] == "GPIB0::8::INSTR-aux-1"
    assert manager.instrs["sr830"][0].aux[4] == "GPIB0::8::INSTR-aux-4"
    assert manager.instrs["sr830"][1].aux[1] == "GPIB0::9::INSTR-aux-1"
    assert manager.instrs["sr830"][1].aux[4] == "GPIB0::9::INSTR-aux-4"
