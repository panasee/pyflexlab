#!/usr/bin/env python
import os
from pathlib import Path
from functools import wraps


from pyomnix.utils import (
    is_notebook,
)

LOCAL_DB_PATH: Path | None = None
OUT_DB_PATH: Path | None = None


def set_envs() -> None:
    """
    set the environment variables from related environment variables
    e.g. PYLAB_DB_LOCAL_XXX -> PYLAB_DB_LOCAL
    """
    for env_var in ["PYLAB_DB_LOCAL", "PYLAB_DB_OUT"]:
        if env_var not in os.environ:
            for key in os.environ:
                if key.startswith(env_var):
                    os.environ[env_var] = os.environ[key]
                    print(f"set with {key}")
                    break
            else:
                print(f"{env_var} not found in environment variables")


def set_paths(
    *, local_db_path: Path | str | None = None, out_db_path: Path | str | None = None
) -> None:
    """
    two ways are provided to set the paths:
    1. set the paths directly in the function (before other modules are imported)
    2. set the paths in the environment variables PYLAB_DB_LOCAL and PYLAB_DB_OUT
    """
    global LOCAL_DB_PATH, OUT_DB_PATH
    if local_db_path is not None:
        LOCAL_DB_PATH = Path(local_db_path)
    else:
        if os.getenv("PYLAB_DB_LOCAL") is None:
            print("PYLAB_DB_LOCAL not set")
        else:
            LOCAL_DB_PATH = Path(os.getenv("PYLAB_DB_LOCAL"))
            print(f"read from PYLAB_DB_LOCAL:{LOCAL_DB_PATH}")

    if out_db_path is not None:
        OUT_DB_PATH = Path(out_db_path)
    else:
        if os.getenv("PYLAB_DB_OUT") is None:
            print("PYLAB_DB_OUT not set")
        else:
            OUT_DB_PATH = Path(os.getenv("PYLAB_DB_OUT"))
            print(f"read from PYLAB_DB_OUT:{OUT_DB_PATH}")


# define constants
SWITCH_DICT = {"on": True, "off": False, "ON": True, "OFF": False}


def handle_keyboard_interrupt(func):
    """##TODO: to add cleanup, now not used"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print("KeyboardInterrupt caught. Cleaning up...")
            # Perform any necessary cleanup here
            return None

    return wrapper


if "__name__" == "__main__":
    if is_notebook():
        print("This is a notebook")
    else:
        print("This is not a notebook")
