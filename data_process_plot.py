#!/usr/bin/env python

import importlib
import matplotlib.pyplot as plt
from file_organizer import FileOrganizer
from measure_manager import MeasureManager
import pandas as pd
import numpy as np
import pltconfig.color_preset as colors
from constants import cm_to_inch

class DataProcessPlot(FileOrganizer):
    """This class is responsible for processing and plotting the data"""
    # define static variables
    legend_font: dict
    """A constant dict used to set the font of the legend in the plot"""

    def __init__(self, proj_name: str, usetex: bool = False, usepgf: bool = False, if_folder_create = True) -> None:
        """
        Initialize the FileOrganizer and load the settings for matplotlib saved in another file
        
        Args:
        - proj_name: the name of the project
        - usetex: whether to use the TeX engine to render text
        - usepgf: whether to use the pgf backend to render text
        - if_folder_create: whether to create the folder for all the measurements in project
        """
        super().__init__(proj_name)
        DataProcessPlot.load_settings(usetex, usepgf)
        self.create_folder("plot")
        if if_folder_create:
            for i in self.query_proj()["measurements"]:
                self.create_folder(f"plot/{i}")

    @staticmethod
    def load_settings(usetex: bool = False, usepgf: bool = False) -> None:
        """load the settings for matplotlib saved in another file"""
        file_name = "pltconfig.plot_config"
        if usetex:
            file_name += "_tex"
            if usepgf:
                file_name += "_pgf"
        else:
            file_name += "_notex"

        config_module = importlib.import_module(f"{file_name}.py")
        DataProcessPlot.legend_font = getattr(config_module, 'legend_font')
