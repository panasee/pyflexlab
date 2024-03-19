#!/usr/bin/env python
import numpy as np

# define constants
cm_to_inch = 0.3937
hplanck = 6.626 * 10**(-34)
hbar = hplanck / 2 / np.pi
hbar_thz = hbar * 10**12
kb = 1.38 * 10**(-23)

def factor(unit:str):
    if unit[0] == "p":
        return 1E12
    if unit[0] == "n":
        return 1E9
    if unit[0] == "u":
        return 1E6
    if unit[0] == "m":
        return 1E3
    if unit[0] == "k":
        return 1E-3
    if unit[0] == "M":
        return 1E-6
    else:
        return 1
