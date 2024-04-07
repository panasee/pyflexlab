#!/usr/bin/env python
"""This module is responsible for processing and plotting the data"""

import importlib
from typing import List, Tuple, Literal
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from common.file_organizer import FileOrganizer
from common.measure_manager import MeasureManager
import common.pltconfig.color_preset as colors
from common.constants import cm_to_inch, factor, default_plot_dict
from datetime import datetime


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

    def load_dfs(self, measurename_all: str, *var_tuple, tmpfolder: str = None) -> None:
        """
        Load a dataframe from a file, save the dataframe as a memeber variable and also return it

        Args:
        - measurename: the measurement name
        - **kwargs: the arguments for the pd.read_csv function
        """
        filepath = self.get_filepath(measurename_all, *var_tuple, tmpfolder=tmpfolder)
        measurename_main, _ = FileOrganizer.measurename_decom(measurename_all)
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
    
    def symmetrize(self, measurename, index_col: any, obj_col: List[any], neutral_point: float = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        do symmetrization to the dataframe and save the symmetric and antisymmetric parts in the original dataframe as new columns 

        Args:
        - index_col: the name of the index column for symmetrization
        - obj_col: a list of the name(s) of the objective column for symmetrization
        - neutral_point: the neutral point for symmetrization
        """
        measurename_main, _ = FileOrganizer.measurename_decom(measurename)
        # Separate the negative and positive parts for interpolation
        df_negative = self.dfs[measurename_main][self.dfs[measurename_main][index_col] < neutral_point][[index_col]+obj_col].copy()
        df_positive = self.dfs[measurename_main][self.dfs[measurename_main][index_col] > neutral_point][[index_col]+obj_col].copy()
        # For symmetrization, we need to flip the negative part and make positions positive
        df_negative[index_col] = -df_negative[index_col]
        # sort them
        df_negative = df_negative.sort_values(by=index_col).reset_index(drop=True)
        df_positive = df_positive.sort_values(by=index_col).reset_index(drop=True)
        # do interpolation for the union of the two parts
        index_union = np.union1d(df_negative[index_col], df_positive[index_col])
        pos_interpolated = np.array([np.interp(index_union, df_positive[index_col], df_positive[obj_col[i]]) for i in range(len(obj_col))])
        neg_interpolated = np.array([np.interp(index_union, df_negative[index_col], df_negative[obj_col[i]]) for i in range(len(obj_col))])
        # Symmetrize and save to DataFrame
        sym = (pos_interpolated + neg_interpolated) / 2
        sym_df = pd.DataFrame(np.transpose(np.append([index_union], sym, axis=0)), columns=[index_col] + [f"{obj_col[i]}sym" for i in range(len(obj_col))])
        antisym = (pos_interpolated - neg_interpolated) / 2
        antisym_df = pd.DataFrame(np.transpose(np.append([index_union], antisym, axis=0)), columns=[index_col] + [f"{obj_col[i]}antisym" for i in range(len(obj_col))])

        return sym_df, antisym_df

    def compare(self, measurename_main: str, columns: List[str], plot_dict: dict = default_plot_dict) -> None:
        ##TODO##
        pass

    @staticmethod
    def time_to_datetime(t : pd.Series, *, past_time: Literal["min", "hour", "no"]="min")-> List[datetime]:
        """
        Convert the time to datetime object, used to split time series without day information

        Args:
        t : pd.Series
            The time series to be converted, format should be like "11:30 PM"
        past_time : Literal["min", "hour", "no"]
            Whether to return the time past from first time points instead of return datetime list 
        Returns:
        List[datetime.datetime]
            The converted datetime object list, year and month are meaningless, just use the date from 1
        """
        datetimes = [datetime.strptime(ts,"%I:%M %p").time() for ts in t]
        day = 1
        datetime_list = []
        tmp = None
        # Iterate over the datetime objects
        for tm in datetimes:
            # Get the date part of the datetime
            if tmp == None:
                pass
            elif tmp > tm:
                day += 1
            tmp = tm
            datetime_list.append(datetime.combine(datetime(1971,9,day),tm))
        if past_time == "no":
            return datetime_list
        else:   
            if past_time == "min":
                factor_time = 60
            if past_time == "hour":
                factor_time = 3600
            return [(t - datetime_list[0]).total_seconds()/factor_time for t in datetime_list]
