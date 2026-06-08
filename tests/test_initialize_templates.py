import pyflexlab
from pyflexlab import constants


def test_initialize_with_templates_warns_and_skips_existing_local_file(monkeypatch, tmp_path):
    monkeypatch.setattr(constants, "LOCAL_DB_PATH", tmp_path)
    existing_notebook = tmp_path / "assist_measure.ipynb"
    existing_notebook.write_text("private local notebook", encoding="utf-8")

    warnings = []

    def collect_warning(message, *args):
        warnings.append(message % args if args else message)

    monkeypatch.setattr(pyflexlab.logger, "warning", collect_warning)

    pyflexlab.initialize_with_templates()

    assert existing_notebook.read_text(encoding="utf-8") == "private local notebook"
    assert any(
        "Skipped existing template: assist_measure.ipynb" in warning
        for warning in warnings
    )
