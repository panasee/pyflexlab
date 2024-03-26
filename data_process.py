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
        self.dfs = {}

    def load_dfs(self, measurename: str, *var_tuple, tmpfolder: str = None) -> None:
        """
        Load a dataframe from a file, save the dataframe as a memeber variable and also return it

        Args:
        - measurename: the measurement name
        - **kwargs: the arguments for the pd.read_csv function
        """
        filepath = self.get_filepath(measurename, *var_tuple, tmpfolder=tmpfolder)
        measurename_main, _ = FileOrganizer.measurename_decom(measurename)
        self.dfs[measurename_main] = pd.read_csv(filepath, sep=r'\s+', skiprows=1, header=None)

    def rename_columns(self, measurename_main: str, columns_name: dict) -> None:
        """
        Rename the columns of the dataframe

        Args:
        - columns: the renaming rules, e.g. {"old_name": "new_name"}
        """
        self.dfs[measurename_main].rename(columns = columns_name, inplace=True)

    @staticmethod
    def merge_with_tolerance(df1: pd.DataFrame, df2: pd.DataFrame, on: any, tolerance: float, suffixes: Tuple[str] = ("_1", "_2")) -> pd.DataFrame:
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
    
    def symmetrize(self, index_col: any, obj_col: List[any]) -> None:
        """
        do symmetrization to the dataframe and save the symmetric and antisymmetric parts in the original dataframe as new columns 

        Args:
        - index_col: the name of the index column for symmetrization
        - obj_col: a list of the name(s) of the objective column for symmetrization
        """
        # Separate the negative and positive parts for interpolation
        df_negative = self.dfs[self.dfs[index_col] < 0][obj_col].copy()
        df_positive = self.dfs[self.dfs[index_col] > 0][obj_col].copy()
        # For symmetrization, we need to flip the negative part and make positions positive
        df_negative[index_col] = -df_negative[index_col]
        # Combine and sort for interpolation, filling gaps in the measurements
        combined = pd.concat([df_negative, df_positive]).sort_values(by=index_col).reset_index(drop=True)
        # Use interpolation to fill in any missing measurements at positions
        interpolated = combined.set_index(index_col).interpolate(method='linear').reset_index()
        # Now, perform the symmetrization: value(positive) - interpolated value at corresponding negative
        # First, ensure every positive position has a corresponding negative one (after interpolation)
        symmetrized_values = []
        antisymmetrized_values = []
        for pos in df_positive[index_col]:
            if pos in interpolated[index_col].values:
                original_value = df_positive[df_positive[index_col] == pos]['Value'].values[0]
                interpolated_value = interpolated[interpolated[index_col] == pos]['Value'].values[0]
                symmetrized_values.append(original_value - interpolated_value)
            else:
                # If there's no matching position after interpolation, this can be handled as needed
                symmetrized_values.append(np.nan)
        # Create a new DataFrame with the symmetrized results
        symmetrized_df = pd.DataFrame({
            index_col: df_positive[index_col],
            'Symmetrized_Value': symmetrized_values
        })

    def compare(self, measurename_main: str, columns: List[str], plot_dict: dict = default_plot_dict) -> None:
        ##TODO##
        pass