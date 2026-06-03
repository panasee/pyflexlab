import pytest

import pyflexlab.equip_wrapper.meters as meters
from pyflexlab.equip_wrapper.meters import WrapperSR860


class FakeSR860Meter:
    def __init__(self):
        self.sine_voltage = 0.0
        self.sine_dc_level = 0.321
        self.frequency = 17.0
        self.dcmode = "DIF"


def make_wrapper():
    wrapper = WrapperSR860.__new__(WrapperSR860)
    wrapper.meter = FakeSR860Meter()
    wrapper.output_target = 0.0
    wrapper.warning_printed = True
    wrapper.info_dict = {
        "frequency": wrapper.meter.frequency,
        "sine_dc_level": wrapper.meter.sine_dc_level,
        "sine_dc_mode": wrapper.meter.dcmode,
    }
    return wrapper


def test_sr860_uni_output_leaves_dc_offset_unchanged_by_default():
    wrapper = make_wrapper()

    wrapper.uni_output("10mV", freq=23, type_str="volt")

    assert wrapper.meter.sine_voltage == pytest.approx(0.01)
    assert wrapper.meter.sine_dc_level == pytest.approx(0.321)
    assert wrapper.meter.frequency == pytest.approx(23)


def test_sr860_uni_output_sets_explicit_dc_offset_and_mode():
    wrapper = make_wrapper()

    wrapper.uni_output("10mV", freq=23, type_str="volt", offset="100mV", dc_mode="COM")

    assert wrapper.meter.sine_voltage == pytest.approx(0.01)
    assert wrapper.meter.sine_dc_level == pytest.approx(0.1)
    assert wrapper.meter.dcmode == "COM"
    assert wrapper.info_dict["sine_dc_level"] == pytest.approx(0.1)
    assert wrapper.info_dict["sine_dc_mode"] == "COM"


def test_sr860_uni_output_allows_explicit_zero_offset():
    wrapper = make_wrapper()

    wrapper.uni_output("10mV", type_str="volt", offset=0)

    assert wrapper.meter.sine_dc_level == 0


@pytest.mark.parametrize("offset", [-5.1, 5.1])
def test_sr860_uni_output_rejects_dc_offset_outside_sr860_range(offset):
    wrapper = make_wrapper()

    with pytest.raises(ValueError, match="between -5 V and 5 V"):
        wrapper.uni_output("10mV", type_str="volt", offset=offset)


def test_sr860_uni_output_rejects_combined_sine_and_dc_level_over_limit():
    wrapper = make_wrapper()

    with pytest.raises(ValueError, match="SR860 sine output limit"):
        wrapper.uni_output("2V", type_str="volt", offset=4)


@pytest.mark.parametrize(
    ("value", "offset", "message"),
    [
        ("10mV", 5.1, "between -5 V and 5 V"),
        ("2V", 4, "SR860 sine output limit"),
    ],
)
def test_sr860_uni_output_uses_logger_raise_error_for_invalid_offset(
    monkeypatch, value, offset, message
):
    wrapper = make_wrapper()
    calls = []

    def fake_raise_error(message_text, exception_type, *args, **kwargs):
        calls.append((message_text, exception_type))
        raise exception_type(message_text)

    monkeypatch.setattr(meters.logger, "raise_error", fake_raise_error)

    with pytest.raises(ValueError, match=message):
        wrapper.uni_output(value, type_str="volt", offset=offset)

    assert calls
    assert calls[0][1] is ValueError
    assert message in calls[0][0]
