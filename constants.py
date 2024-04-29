#!/usr/bin/env python
import sys
import numpy as np
import common.pltconfig.color_preset as colors

# define constants
cm_to_inch = 0.3937
hplanck = 6.626 * 10**(-34)
hbar = hplanck / 2 / np.pi
hbar_thz = hbar * 10**12
kb = 1.38 * 10**(-23)
unit_factor_fromSI= {"p":1E12, "n":1E9, "u":1E6, "m":1E3, "k":1E-3, "M":1E-6, "G":1E-9}
unit_factor_toSI= {"p":1E-12, "n":1E-9, "u":1E-6, "m":1E-3, "k":1E3, "M":1E6, "G":1E9}

#define plotting default settings
default_plot_dict = {"color":colors.Genshin["Nilou"][0], "linewidth":1, "linestyle":"-", "marker":"o", "markersize":2, "markerfacecolor":"None", "markeredgecolor":"black", "markeredgewidth":0.3, "label":"", "alpha":0.77}

def factor(unit:str, mode: str = "from_SI"):
    """
    Transform the SI unit to targeted unit or in the reverse order.

    Args:
    unit: str
        The unit to be transformed.
    mode: str
        The direction of the transformation. "from_SI" means transforming from SI unit to the targeted unit, and "to_SI" means transforming from the targeted unit to SI unit.
    """
    if mode == "from_SI":
        if unit[0] in unit_factor_fromSI:
            return unit_factor_fromSI.get(unit[0])
        else:
            return 1
    if mode == "to_SI":
        if unit[0] in unit_factor_fromSI:
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

if "__name__" == "__main__":
    if is_notebook():
        print("This is a notebook")
    else:
        print("This is not a notebook")