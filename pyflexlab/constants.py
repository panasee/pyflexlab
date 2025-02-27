#!/usr/bin/env python
import re
import os
from pathlib import Path
import sys
from datetime import datetime
from functools import wraps
from typing import Literal, Generator, Sequence, Optional
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib import colormaps

import numpy as np
import pandas as pd

import pyflexlab.pltconfig.color_preset as colors

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

def set_paths(*, local_db_path: Path | str | None = None, out_db_path: Path | str | None = None) -> None:
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
cm_to_inch = 0.3937
hplanck = 6.626 * 10 ** (-34)
hbar = hplanck / 2 / np.pi
hbar_thz = hbar * 10 ** 12
kb = 1.38 * 10 ** (-23)
unit_factor_fromSI = {"": 1, "f": 1E15, "p": 1E12, "n": 1E9, "u": 1E6, "m": 1E3, "k": 1E-3, "M": 1E-6, "G": 1E-9,
                      "T": 1E-12,
                      "P": 1E-15}
unit_factor_toSI = {"": 1, "f": 1E-15, "p": 1E-12, "n": 1E-9, "u": 1E-6, "m": 1E-3, "k": 1E3, "M": 1E6, "G": 1E9,
                    "T": 1E12,
                    "P": 1E15}

#define plotting default settings
default_plot_dict = {"color": colors.Presets["Nl"][0], "linewidth": 1, "linestyle": "-", "marker": "o",
                     "markersize": 1.5, "markerfacecolor": "None", "markeredgecolor": "black", "markeredgewidth": 0.3,
                     "label": "", "alpha": 0.77}

switch_dict = {"on": True, "off": False, "ON": True, "OFF": False}


def factor(unit: str, mode: str = "from_SI"):
    """
    Transform the SI unit to targeted unit or in the reverse order.

    Args:
    unit: str
        The unit to be transformed.
    mode: str
        The direction of the transformation. "from_SI" means transforming from SI unit to the targeted unit, and "to_SI" means transforming from the targeted unit to SI unit.
    """
    # add judgement for the length to avoid m (meter) T (tesla) to be recognized as milli
    if len(unit) <= 1:
        return 1
    if mode == "from_SI":
        if unit[0] in unit_factor_fromSI:
            return unit_factor_fromSI.get(unit[0])
        else:
            return 1
    if mode == "to_SI":
        if unit[0] in unit_factor_toSI:
            return unit_factor_toSI.get(unit[0])
        else:
            return 1


def is_notebook() -> bool:
    """
    judge if the code is running in a notebook environment.
    """
    if 'ipykernel' in sys.modules and 'IPython' in sys.modules:
        try:
            from IPython import get_ipython
            if 'IPKernelApp' in get_ipython().config:
                return True
        except:
            pass
    return False


def split_no_str(s: str | int | float) -> tuple[float | None, str | None]:
    """
    split the string into the string part and the float part.

    Args:
        s (str): the string to split

    Returns:
        tuple[float,str]: the string part and the integer part
    """
    if isinstance(s, (int, float)):
        return s, ""
    match = re.match(r"([+-]?[0-9.]+)([a-zA-Z]*)", s, re.I)

    if match:
        items = match.groups()
        return float(items[0]), items[1]
    else:
        return None, None


def convert_unit(before: float | int | str | list[float | int | str, ...] | tuple[float | int | str, ...] | np.ndarray,
                 target_unit: str = "") -> tuple[float, str] | tuple[list[float], list[str]]:
    """
    Convert the value with the unit to the SI unit.

    Args:
        before (float | str): the value with the unit
        target_unit (str): the target unit

    Returns:
        tuple[float, str]: the value in the target unit and the whole str with final unit
    """
    if isinstance(before, (int, float, str)):
        value, unit = split_no_str(before)
        value_SI = value * factor(unit, mode="to_SI")
        new_value = value_SI * factor(target_unit, mode="from_SI")
        return new_value, f"{new_value}{target_unit}"
    elif isinstance(before, (np.integer, np.floating)):
        return convert_unit(float(before), target_unit)
    elif isinstance(before, (list, tuple, np.ndarray)):
        return [convert_unit(i, target_unit)[0] for i in before], [convert_unit(i, target_unit)[1] for i in before]


def print_progress_bar(iteration: float, total: float, prefix='', suffix='', decimals=1, length=50, fill='#',
                       print_end="\r") -> None:
    """
    Call in a loop to create terminal progress bar

    Args:
        iteration (float): current iteration
        total (float): total iterations
        prefix (str): prefix string
        suffix (str): suffix string
        decimals (int): positive number of decimals in percent complete
        length (int): character length of bar
        fill (str): bar fill character
        print_end (str): end character (e.g. "\r", "\r\n")
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    barr = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} [{barr}] {percent}% {suffix}', end=print_end, flush=True)
    # Print New Line on Complete
    if iteration == total:
        print()


def gen_seq(start, end, step):
    """
    double-ended bi-direction sequence generator
    """
    if step == 0:
        raise ValueError("step should not be zero")
    if step * (end - start) < 0:
        step *= -1
    value = start
    while (value - end) * step < 0:
        yield value
        value += step
    yield end


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


def constant_generator(value, repeat: int | Literal["inf"] = "inf"):
    """
    generate a constant value infinitely
    """
    if repeat == "inf":
        while True:
            yield value
    else:
        idx = 0
        while idx < repeat:
            idx += 1
            yield value


def time_generator(format_str: str = "%Y-%m-%d_%H:%M:%S"):
    """
    generate current time always

    Args:
        format_str (str): the format of the time
    """
    while True:
        yield datetime.now().isoformat(sep="_", timespec="milliseconds")


def combined_generator_list(lst_gens: list[Generator]):
    """
    combine a list of generators into one generator generating a whole list
    """
    while True:
        try:
            list_ini = [next(i) for i in lst_gens]
            list_fin = []
            for i in list_ini:
                if isinstance(i, list | tuple):
                    list_fin.extend(i)
                else:
                    list_fin.append(i)
            yield list_fin
        except StopIteration:
            break


def next_lst_gen(lst_gens: list[Generator]):
    """
    get the next value of the generators in the list ONCE
    """
    try:
        list_ini = [next(i) for i in lst_gens]
        list_fin = []
        for i in list_ini:
            if isinstance(i, list | tuple):
                list_fin.extend(i)
            else:
                list_fin.append(i)
        return list_fin
    except StopIteration:
        return None


def rename_duplicates(columns: list[str]) -> list[str]:
    """
    rename the duplicates with numbers (like ["V","V"] to ["V1","V2"])
    """
    count_dict = {}
    renamed_columns = []
    for col in columns:
        if col in count_dict:
            count_dict[col] += 1
            renamed_columns.append(f"{col}{count_dict[col]}")
        else:
            count_dict[col] = 1
            renamed_columns.append(col)
    return renamed_columns


def hex_to_rgb(hex_str: str, fractional: bool = True) -> tuple[int, ...] | tuple[float, ...]:
    """
    convert hex color to rgb color

    Args:
        hex_str (str): hex color
        fractional (bool): if the return value is fractional or not
    """
    hex_str = hex_str.lstrip('#')
    if fractional:
        return tuple(int(hex_str[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))


def timestr_convert(t: pd.Series | Sequence[str] | np.ndarray, format_str: str = "%Y-%m-%d_%H:%M:%S.%f", *,
                    elapsed: Optional[Literal["sec", "min", "hour"]] = None) -> list[datetime] | list[float]:
    """
    Convert the time to datetime object, used to split time series without day information

    Args:
    t : pd.Series
        The time series to be converted, format should be like "11:30 PM"
    format_str : str
        The format string for the time, e.g. "%I:%M %p"
        the meaning of each character and optional characters is as follows:
        %H : Hour (24-hour clock) as a zero-padded decimal number. 00, 01, ..., 23
        %I : Hour (12-hour clock) as a zero-padded decimal number. 01, 02, ..., 12
        %p : Locale's equivalent of either AM or PM.
        %M : Minute as a zero-padded decimal number. 00, 01, ..., 59
        %S : Second as a zero-padded decimal number. 00, 01, ..., 59
        %f : Microsecond as a decimal number, zero-padded on the left. 000000, 000001, ..., 999999
        %a : Weekday as locale's abbreviated name.
        %A : Weekday as locale's full name.
        %w : Weekday as a decimal number, where 0 is Sunday and 6 is Saturday.
        %d : Day of the month as a zero-padded decimal number. 01, 02, ..., 31
        %b : Month as locale's abbreviated name.
        %B : Month as locale's full name.
        %m : Month as a zero-padded decimal number. 01, 02, ..., 12
        %y : Year without century as a zero-padded decimal number. 00, 01, ..., 99
        %Y : Year with century as a decimal number. 0001, 0002, ..., 2013, 2014, ..., 9998, 9999
    elapsed : Literal["sec", "min", "hour"]
        Whether to return the time past from first time points instead of return datetime list
    Returns:
    list[datetime] | list[float]
        The datetime list or the time past from the first time points
    """
    datetime_lst = [datetime.strptime(ts, format_str) for ts in t]
    if not datetime_lst:
        raise ValueError("The input time series is empty")
    if elapsed is not None:
        time_start = datetime_lst[0]
        match elapsed:
            case "sec":
                elapsed_times = [(dt - time_start).total_seconds() for dt in datetime_lst]
            case "min":
                elapsed_times = [(dt - time_start).total_seconds() / 60 for dt in datetime_lst]
            case "hour":
                elapsed_times = [(dt - time_start).total_seconds() / 3600 for dt in datetime_lst]
            case _:
                raise ValueError("The elapsed argument should be one of 'sec', 'min', 'hour'")
        return elapsed_times
    else:
        return datetime_lst


def truncate_cmap(cmap, min_val: float = 0.0, max_val: float = 1.0, n: int = 256):
    """
    truncate the colormap to the specific range

    Args:
        cmap : LinearSegmentedColormap | ListedColormap
            the colormap to be truncated
        min_val : float
            the minimum value of the colormap
        max_val : float
            the maximum value of the colormap
        n : int
            the number of colors in the colormap
    """
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{min_val:.2f},{max_val:.2f})", cmap(np.linspace(min_val, max_val, n)))
    return new_cmap


def combine_cmap(cmap_lst: list, segment: int = 128):
    """
    combine the colormaps in the list

    Args:
        cmap_lst : list
            the list of colormaps to be combined
        segment : int
            the number of segments in each colormap
    """
    c_lst = []
    for cmap in cmap_lst:
        c_lst.extend(cmap(np.linspace(0, 1, segment)))
    new_cmap = LinearSegmentedColormap.from_list("combined", c_lst)
    return new_cmap



if "__name__" == "__main__":
    if is_notebook():
        print("This is a notebook")
    else:
        print("This is not a notebook")
