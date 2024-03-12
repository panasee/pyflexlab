#!/usr/bin/env python

import importlib
import matplotlib
import matplotlib.pyplot as plt
from common.file_organizer import FileOrganizer
from common.measure_manager import MeasureManager
import pandas as pd
import numpy as np
import common.pltconfig.color_preset as colors
from common.constants import cm_to_inch
from typing import List

class DataPlot(FileOrganizer):
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
        DataPlot.load_settings(usetex, usepgf)
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
        DataPlot.legend_font = getattr(config_module, 'legend_font')

    @staticmethod
    def merge_legends(*ax_list: List[matplotlib.axes.Axes],loc: str = "best") -> None:
        """
        Merge and show the legends of multiple axes

        Args:
        - ax_list: a list of axes
        """
        handles, labels = zip(*[ax.get_legend_handles_labels() for ax in ax_list])
        ax_list[0].legend(handles=handles, labels=labels, loc=loc, prop=DataPlot.legend_font)


class DataProcess(FileOrganizer):
    """This class is responsible for processing the data"""
    def __init__(self, proj_name: str) -> None:
        """
        Initialize the FileOrganizer and load the settings for matplotlib saved in another file
        
        Args:
        - proj_name: the name of the project
        """
        super().__init__(proj_name)

    @staticmethod
    def merge_with_tolerance(df1: pd.DataFrame, df2: pd.DataFrame, on: any, tolerance: float, suffixes = ("_1", "_2")) -> pd.DataFrame:
        """
        Merge two dataframes with tolerance

        Args:
        - df1: the first dataframe
        - df2: the second dataframe
        - on: the column to merge on
        - tolerance: the tolerance for the merge
        - suffixes: the suffixes for the columns of the two dataframes
        """
        df1 = df1.sort_values(by=on).reset_index(drop=True)
        df2 = df2.sort_values(by=on).reset_index(drop=True)

        i = 0
        j = 0

        result = []

        while i < len(df1) and j < len(df2):
            if abs(df1.loc[i, on] - df2.loc[j, on]) <= tolerance:
                row = pd.concat([df1.loc[i].add_suffix(suffixes[0]), df2.loc[j].add_suffix(suffixes[1])])
                result.append(row)
                i += 1
                j += 1
            elif df1.loc[i, on] < df2.loc[j, on]:
                i += 1
            else:
                j += 1

        return pd.DataFrame(result)
