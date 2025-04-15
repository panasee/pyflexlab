"""Python package for flexible laboratory data analysis and visualization."""

from pyomnix.omnix_logger import get_logger
from pyflexlab.constants import set_envs, set_paths

set_envs()
set_paths()

from pyflexlab.file_organizer import FileOrganizer
from pyflexlab.measure_manager import MeasureManager
from pyflexlab.data_process import DataProcess
from pyflexlab.measure_flow import MeasureFlow

logger = get_logger(__name__)

def initialize_with_templates():
    """
    This function will copy necessary files from the templates folder to the LOCAL_DB_PATH
    """
    import os
    import shutil
    from pathlib import Path
    from pyflexlab.constants import LOCAL_DB_PATH

    if LOCAL_DB_PATH is None:
        logger.warning("LOCAL_DB_PATH is not set. Cannot copy templates.")
        return

    local_db_dir = Path(LOCAL_DB_PATH)
    local_db_dir.mkdir(parents=True, exist_ok=True)

    templates_dir = Path(__file__).parent / "templates"

    for template_file in templates_dir.glob("**/*"):
        if template_file.is_file():
            # Calculate relative path from templates directory
            rel_path = template_file.relative_to(templates_dir)
            target_path = local_db_dir / rel_path

            # Create parent directories if needed
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy only if the file doesn't exist
            if not target_path.exists():
                shutil.copy2(template_file, target_path)
                logger.info(f"Copied template: {rel_path}")
