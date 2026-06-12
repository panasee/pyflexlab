import datetime
import json
import shutil
from pathlib import Path

import pytest

from pyflexlab.file_organizer import FileOrganizer
from pyflexlab.equip_wrapper.meters import SourceMeter
from pyflexlab.measure_manager import MeasureManager
from pyflexlab.recipe_builders import MeasureModules


class FakeBiasedSource(SourceMeter):
    def __init__(self):
        super().__init__()
        self.safe_step = 1e-3
        self.calls = []
        self.meter = "fake-biased-source"

    def setup(self, function, *vargs, source_type=None, sense_type=None, **kwargs):
        pass

    def info_sync(self):
        pass

    @property
    def sense_range_volt(self):
        return 1

    @sense_range_volt.setter
    def sense_range_volt(self, value):
        pass

    @property
    def sense_range_curr(self):
        return 1

    @sense_range_curr.setter
    def sense_range_curr(self, value):
        pass

    def sense(self, type_str="volt"):
        return 0

    def output_switch(self, switch):
        self.info_dict["output_status"] = bool(switch)

    def uni_output(self, value, **kwargs):
        self.calls.append((value, kwargs))
        return value

    def get_output_status(self):
        return 0, 0

    def shutdown(self):
        self.output_switch(False)


@pytest.fixture
def manager(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    local_db = repo_root / "pyflexlab" / "templates"
    out_db = tmp_path / "out"
    out_db.mkdir()

    old_local_db = FileOrganizer._local_database_dir
    old_out_db = FileOrganizer._out_database_dir
    old_trash_dir = FileOrganizer._trash_dir
    old_measure_types = FileOrganizer.measure_types_json
    old_proj_rec = FileOrganizer.proj_rec_json

    FileOrganizer.reload_paths(local_db_path=local_db, out_db_path=out_db)
    FileOrganizer.measure_types_json = json.loads(
        (local_db / "measure_types.json").read_text(encoding="utf-8")
    )
    FileOrganizer.proj_rec_json = {
        "BiasedSource": {
            "created_date": "2026-06-11",
            "last_modified": "2026-06-11",
            "measurements": [],
            "plan": {},
        }
    }
    (out_db / "project_record.json").write_text(
        json.dumps(FileOrganizer.proj_rec_json),
        encoding="utf-8",
    )

    test_manager = object.__new__(MeasureManager)
    test_manager.proj_name = "BiasedSource"
    test_manager.today = datetime.date(2026, 6, 11)
    test_manager._out_database_dir = out_db
    test_manager._out_database_dir_proj = out_db / "BiasedSource"
    test_manager._out_database_dir_proj.mkdir()
    test_manager._csv_fast_writer = None
    test_manager.df_cache = None

    yield test_manager

    MeasureManager.record_finalize(test_manager)
    FileOrganizer._local_database_dir = old_local_db
    FileOrganizer._out_database_dir = old_out_db
    FileOrganizer._trash_dir = old_trash_dir
    FileOrganizer.measure_types_json = old_measure_types
    FileOrganizer.proj_rec_json = old_proj_rec
    if out_db.exists():
        shutil.rmtree(out_db)


def test_biased_source_modules_are_parsed_as_source_modes(manager):
    mainname, _, details = FileOrganizer.name_fstr_gen(
        "I_source_biased_ac", require_detail=True
    )

    assert mainname == "I--"
    assert details == [
        {"sweep_fix": "biased", "ac_dc": "ac", "source_sense": "source"}
    ]

    sources, senses, externals = MeasureManager.extract_info_mods(
        ("I_source_biased_ac",),
        "10nA",
        17,
        "1uA",
        "100nA",
        1,
        0,
        "0-max-0",
    )

    assert not senses
    assert not externals
    assert sources[0]["I"] == {
        "sweep_fix": "biased",
        "ac_dc": "ac",
        "fix": "10nA",
        "max": "1uA",
        "step": "100nA",
        "mode": "0-max-0",
        "freq": 17,
    }


def test_biased_source_record_columns_include_ac_and_bias(manager):
    file_path, record_num, _ = MeasureManager.record_init(
        manager,
        ("V_source_biased_ac", "V_sense_ac"),
        "10mV",
        17,
        "100mV",
        "10mV",
        1,
        0,
        "0-max-0",
        "",
        2,
        0,
        use_fast_writer=False,
    )

    assert record_num == 7
    assert file_path.read_text(encoding="utf-8").splitlines()[0] == (
        "time,V_ac_source,V_bias,X,Y,R,Theta"
    )


def test_source_biased_apply_sweeps_offset_and_records_ac_and_bias(manager):
    source = FakeBiasedSource()

    gen = MeasureManager.source_biased_apply(
        manager,
        "I",
        "10nA",
        source,
        max_value="20nA",
        step_value="20nA",
        compliance="1V",
        freq=17,
        sweepmode="0-max-0",
        source_wait=0,
    )

    assert next(gen) == pytest.approx((10e-9, 0.0))
    assert next(gen) == pytest.approx((10e-9, 20e-9))
    assert next(gen) == pytest.approx((10e-9, 20e-9))
    assert next(gen) == pytest.approx((10e-9, 0.0))
    assert source.calls == [
        (
            pytest.approx(10e-9),
            {
                "freq": pytest.approx(17),
                "compliance": pytest.approx(1),
                "type_str": "curr",
                "offset": pytest.approx(0.0),
            },
        ),
        (
            pytest.approx(10e-9),
            {
                "freq": pytest.approx(17),
                "compliance": pytest.approx(1),
                "type_str": "curr",
                "offset": pytest.approx(20e-9),
            },
        ),
        (
            pytest.approx(10e-9),
            {
                "freq": pytest.approx(17),
                "compliance": pytest.approx(1),
                "type_str": "curr",
                "offset": pytest.approx(20e-9),
            },
        ),
        (
            pytest.approx(10e-9),
            {
                "freq": pytest.approx(17),
                "compliance": pytest.approx(1),
                "type_str": "curr",
                "offset": pytest.approx(0.0),
            },
        ),
    ]


def test_recipe_builder_exposes_biased_ac_source_modules():
    meter = object()

    current_module = MeasureModules.biased_current_source(
        "10nA",
        "1uA",
        "100nA",
        freq=17,
        high=1,
        low=0,
        sweepmode="0-max-0",
        meter=meter,
        compliance="1V",
    )
    voltage_module = MeasureModules.biased_voltage_source(
        "10mV",
        "100mV",
        "10mV",
        freq=17,
        high=1,
        low=0,
        sweepmode="0-max-0",
        meter=meter,
        compliance=None,
    )

    assert current_module.module_id == "source.biased_current"
    assert current_module.measure_mod == "I_source_biased_ac"
    assert current_module.args == ("10nA", 17, "1uA", "100nA", 1, 0, "0-max-0")
    assert current_module.wrapper is meter
    assert current_module.compliance == "1V"

    assert voltage_module.module_id == "source.biased_voltage"
    assert voltage_module.measure_mod == "V_source_biased_ac"
    assert voltage_module.args == ("10mV", 17, "100mV", "10mV", 1, 0, "0-max-0")
