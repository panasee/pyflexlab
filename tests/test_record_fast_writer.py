import datetime
import json
import shutil
import unittest
from pathlib import Path

from pyflexlab.file_organizer import FileOrganizer
from pyflexlab.measure_manager import MeasureManager


class RecordFastWriterTest(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.tmp_root = self.repo_root / ".test_tmp_record_fast_writer"
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)

        self.local_db = self.repo_root / "pyflexlab" / "templates"
        self.out_db = self.tmp_root / "out"
        self.out_db.mkdir(parents=True)

        self._old_local_db = FileOrganizer._local_database_dir
        self._old_out_db = FileOrganizer._out_database_dir
        self._old_trash_dir = FileOrganizer._trash_dir
        self._old_measure_types = FileOrganizer.measure_types_json
        self._old_proj_rec = FileOrganizer.proj_rec_json

        FileOrganizer.reload_paths(
            local_db_path=self.local_db,
            out_db_path=self.out_db,
        )
        FileOrganizer.measure_types_json = json.loads(
            (self.local_db / "measure_types.json").read_text(encoding="utf-8")
        )
        FileOrganizer.proj_rec_json = {
            "HeaderBug": {
                "created_date": "2026-05-04",
                "last_modified": "2026-05-04",
                "measurements": [],
                "plan": {},
            }
        }
        (self.out_db / "project_record.json").write_text(
            json.dumps(FileOrganizer.proj_rec_json),
            encoding="utf-8",
        )

        self.manager = object.__new__(MeasureManager)
        self.manager.proj_name = "HeaderBug"
        self.manager.today = datetime.date(2026, 5, 4)
        self.manager._out_database_dir = self.out_db
        self.manager._out_database_dir_proj = self.out_db / "HeaderBug"
        self.manager._out_database_dir_proj.mkdir()
        self.manager._csv_fast_writer = None
        self.manager.df_cache = None

    def tearDown(self):
        MeasureManager.record_finalize(self.manager)
        FileOrganizer._local_database_dir = self._old_local_db
        FileOrganizer._out_database_dir = self._old_out_db
        FileOrganizer._trash_dir = self._old_trash_dir
        FileOrganizer.measure_types_json = self._old_measure_types
        FileOrganizer.proj_rec_json = self._old_proj_rec
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)

    def test_fast_record_init_overwrites_existing_csv(self):
        args = (
            ("I_source_sweep_dc", "V_sense_dc", "T_vary"),
            1,
            0.5,
            1,
            1,
            "0-1",
            "",
            1,
            1,
            300,
            301,
        )

        file_path, record_num, _ = MeasureManager.record_init(
            self.manager, *args, use_fast_writer=True
        )
        MeasureManager.record_update(
            self.manager,
            file_path,
            record_num,
            tuple(["t1"] + [1] * (record_num - 1)),
            force_write=True,
        )
        MeasureManager.record_finalize(self.manager)

        file_path, record_num, _ = MeasureManager.record_init(
            self.manager, *args, use_fast_writer=True
        )
        MeasureManager.record_update(
            self.manager,
            file_path,
            record_num,
            tuple(["t2"] + [2] * (record_num - 1)),
            force_write=True,
        )
        MeasureManager.record_finalize(self.manager)

        self.assertEqual(
            file_path.read_text(encoding="utf-8").splitlines(),
            [
                "time,I_source,V,T",
                "t2,2.000000000000,2.000000000000,2.000000000000",
            ],
        )

    def test_fast_record_init_backup_includes_buffered_rows(self):
        args = (
            ("I_source_sweep_dc", "V_sense_dc", "T_vary"),
            1,
            0.5,
            1,
            1,
            "0-1",
            "",
            1,
            1,
            300,
            301,
        )

        file_path, record_num, _ = MeasureManager.record_init(
            self.manager, *args, use_fast_writer=True
        )
        MeasureManager.record_update(
            self.manager,
            file_path,
            record_num,
            tuple(["t1"] + [1] * (record_num - 1)),
        )

        file_path, record_num, _ = MeasureManager.record_init(
            self.manager, *args, use_fast_writer=True
        )
        MeasureManager.record_update(
            self.manager,
            file_path,
            record_num,
            tuple(["t2"] + [2] * (record_num - 1)),
            force_write=True,
        )
        MeasureManager.record_finalize(self.manager)

        backup_path = file_path.parent / f"{file_path.name}.bak"
        self.assertEqual(
            backup_path.read_text(encoding="utf-8").splitlines(),
            [
                "time,I_source,V,T",
                "t1,1.000000000000,1.000000000000,1.000000000000",
            ],
        )


if __name__ == "__main__":
    unittest.main()
