#!/usr/bin/env python
import os
from pathlib import Path
from functools import wraps

from pyomnix.omnix_logger import get_logger
from pyomnix.utils import (
    is_notebook,
)

logger = get_logger(__name__)

LOCAL_DB_PATH: Path | None = None
OUT_DB_PATH: Path | None = None


class BoundedCounter:
    def __init__(self, value: int = 0, min_val: int = 0, max_val: int = 10):
        self._value = value
        self.min_val = min_val
        self.max_val = max_val
        self._apply_bounds()

    def _apply_bounds(self):
        """Clamp the value between min_val and max_val."""
        if self.max_val is not None:
            self._value = min(self._value, self.max_val)
        self._value = max(self._value, self.min_val)

    @property
    def value(self):
        """Get the current counter value."""
        return self._value

    @value.setter
    def value(self, new_value):
        """Set the counter value (automatically clamped)."""
        self._value = new_value
        self._apply_bounds()

    def count_down(self) -> bool:
        """Count down the counter value and takes 1 second,
        if the counter has reached the min value, return False, otherwise return True"""
        if self._value == self.min_val:
            return False
        self._value -= 1
        self._apply_bounds()
        return True

    def reset_to_max(self):
        """Reset the counter value to the max value"""
        self._value = self.max_val

    # Make it behave like an integer in expressions
    def __int__(self):
        return self._value

    def __float__(self):
        return float(self._value)

    # Operator overloading (+, -, +=, -=)
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return BoundedCounter(self._value + other, self.min_val, self.max_val)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return BoundedCounter(self._value - other, self.min_val, self.max_val)
        return NotImplemented

    def __iadd__(self, other):
        if isinstance(other, (int, float)):
            self._value += other
            self._apply_bounds()
            return self
        return NotImplemented

    def __isub__(self, other):
        if isinstance(other, (int, float)):
            self._value -= other
            self._apply_bounds()
            return self
        return NotImplemented

    def __repr__(self):
        return f"BoundedCounter({self._value}, min={self.min_val}, max={self.max_val})"

    def __str__(self):
        return str(self._value)


class SafePath(Path):
    """
    A path class that automatically removes leading slashes or backslashes from the other argument to avoid unexpected behavior caused by absolute path concatenation
    """

    def __truediv__(self, other: str) -> Path:
        if isinstance(other, (str, Path)):
            other_str = str(other)
            if other_str.startswith(("/", "\\")):
                other = other_str.lstrip("/\\")
        return super().__truediv__(other)


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
                    logger.info(f"set with {key}")
                    break
            else:
                logger.warning(f"{env_var} not found in environment variables")


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
            logger.warning("PYLAB_DB_LOCAL not set")
        else:
            LOCAL_DB_PATH = Path(os.getenv("PYLAB_DB_LOCAL"))
            logger.info(f"read from PYLAB_DB_LOCAL:{LOCAL_DB_PATH}")

    if out_db_path is not None:
        OUT_DB_PATH = Path(out_db_path)
    else:
        if os.getenv("PYLAB_DB_OUT") is None:
            logger.warning("PYLAB_DB_OUT not set")
        else:
            OUT_DB_PATH = Path(os.getenv("PYLAB_DB_OUT"))
            logger.info(f"read from PYLAB_DB_OUT:{OUT_DB_PATH}")


# define constants
SWITCH_DICT = {"on": True, "off": False, "ON": True, "OFF": False}


def handle_keyboard_interrupt(func):
    """##TODO: to add cleanup, now not used"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt caught. Cleaning up...")
            # Perform any necessary cleanup here
            return None

    return wrapper


if "__name__" == "__main__":
    if is_notebook():
        logger.info("This is a notebook")
    else:
        logger.info("This is not a notebook")
