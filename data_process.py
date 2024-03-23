#!/usr/bin/env python
"""This module is responsible for processing and plotting the data"""

import importlib
from typing import List, Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from common.file_organizer import FileOrganizer
from common.measure_manager import MeasureManager
import common.pltconfig.color_preset as colors
from common.constants import cm_to_inch, factor, default_plot_dict


class DataProcess(FileOrganizer):
    """This class is responsible for processing the data"""
    def __init__(self, proj_name: str) -> None:
        """
        Initialize the FileOrganizer and load the settings for matplotlib saved in another file
        
        Args:
        - proj_name: the name of the project
        """
        super().__init__(proj_name)
        self.df = None

    def load_df(self, measurename: str, *var_tuple, tmpfolder: str = None) -> pd.DataFrame:
        """
        Load a dataframe from a file, save the dataframe as a memeber variable and also return it

        Args:
        - measurename: the measurement name
        - **kwargs: the arguments for the pd.read_csv function
        """
        filepath = self.get_filepath(measurename, *var_tuple, tmpfolder)
        self.df = pd.read_csv(filepath, sep=r'\s+', skiprows=1, header=None)
        return self.df

    def rename_columns(self, columns: dict) -> None:
        """
        Rename the columns of the dataframe

        Args:
        - columns: the renaming rules, e.g. {"old_name": "new_name"}
        """
        self.df.rename(columns = columns, inplace=True)

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
