#!/usr/bin/env python
"""This module is responsible for processing and plotting the data"""
from itertools import groupby
from os.path import commonpath
from typing import Literal, Sequence, List, Optional
import numpy as np
import pandas as pd

from pylab_dk.file_organizer import FileOrganizer, print_help_if_needed
from pylab_dk.constants import factor
from datetime import datetime, timedelta


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

    @print_help_if_needed
    def load_dfs(self, measure_mods: tuple[str], *var_tuple: float | str, tmpfolder: str = None, cached: bool = False,
                 header: Literal[None, "infer"] = "infer", skiprows: int = None) -> pd.DataFrame:
        """
        Load a dataframe from a file, save the dataframe as a member variable and also return it

        Args:
        - measure_mods: the measurement modules
        - *var_tuple: the arguments for the modules
        - **kwargs: the arguments for the pd.read_csv function
        - cached: whether to save the df into self.dfs["cache"] instead of self.dfs (overwritten by the next load_dfs call, only with temperary usage)
        """
        file_path = self.get_filepath(measure_mods, *var_tuple, tmpfolder=tmpfolder)
        mainname_str, _ = FileOrganizer.name_fstr_gen(*measure_mods)
        if not cached:
            self.dfs[mainname_str] = pd.read_csv(file_path, sep=',', skiprows=skiprows, header=header,
                                                 float_precision='round_trip')
            return self.dfs[mainname_str].copy()
        else:
            self.dfs["cache"] = pd.read_csv(file_path, sep=',', skiprows=skiprows, header=header,
                                            float_precision='round_trip')
            return self.dfs["cache"].copy()

    def rename_columns(self, measurename_main: str, rename_dict: dict) -> None:
        """
        Rename the columns of the dataframe

        Args:
        - rename_dict: the renaming rules, e.g. {"old_name": "new_name"}
        """
        self.dfs[measurename_main].rename(columns=rename_dict, inplace=True)
        if "cache" in self.dfs:
            self.dfs["cache"].rename(columns=rename_dict, inplace=True)

    @staticmethod
    def merge_with_tolerance(df1: pd.DataFrame, df2: pd.DataFrame, on: any, tolerance: float, suffixes: tuple[str] = ("_1", "_2")) -> pd.DataFrame:
        """
        Merge two dataframes with tolerance, unmatched rows will be dropped

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

        return pd.DataFrame(result).copy()

    @staticmethod
    def symmetrize(ori_df: pd.DataFrame, index_col: str | float | int,
                   obj_col: list[str | float | int], neutral_point: float = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        do symmetrization to the dataframe w.r.t. the index col and return the symmetric and antisymmetric DataFrames,
        note that this function is dealing with only one dataframe, meaning the positive and negative parts
        are to be combined first (no need to sort)
        e.g. idx col is [-1,-2,-3,0,4,2,1], obj cols corresponding to -1 will add/minus the corresponding obj cols of 1
            that of -3 will be added/minus that interpolated by 2 and 4, etc. (positive - negative)/2 for antisym

        Args:
        - ori_df: the original dataframe
        - index_col: the name of the index column for symmetrization
        - obj_col: a list of the name(s) of the objective column for symmetrization
        - neutral_point: the neutral point for symmetrization

        Returns:
        - pd.DataFrame[0]: the symmetric part (col names are suffixed with "_sym")
        - pd.DataFrame[1]: the antisymmetric part (col names are suffixed with "_antisym")
        """
        # Separate the negative and positive parts for interpolation
        df_negative = ori_df[ori_df[index_col] < neutral_point][[index_col]+obj_col].copy()
        df_positive = ori_df[ori_df[index_col] > neutral_point][[index_col]+obj_col].copy()
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
        sym_df = pd.DataFrame(np.transpose(np.append([index_union], sym, axis=0)), columns=[index_col] + [f"{obj_col[i]}_sym" for i in range(len(obj_col))])
        antisym = (pos_interpolated - neg_interpolated) / 2
        antisym_df = pd.DataFrame(np.transpose(np.append([index_union], antisym, axis=0)), columns=[index_col] + [f"{obj_col[i]}_antisym" for i in range(len(obj_col))])

        #return pd.concat([sym_df, antisym_df], axis = 1)
        return sym_df, antisym_df

    @staticmethod
    def difference(ori_df: pd.DataFrame | Sequence[pd.DataFrame],
                   index_col: str | float | int | Sequence[str | float | int],
                   target_col: str | float | int | Sequence[str | float | int],
                   relative: bool = False) -> pd.DataFrame:
        """
        Calculate the difference between the values in the column(should have the same name) of two dataframes
        or in two columns of one dataframe, the final df will use the first col name given
        NOTE the interpolation will cause severe error for extension outside the original range
        the overlapped values will be AVERAGED

        Args:
        - ori_df: the original dataframe(s)
        - index_col: the name of the index column for symmetrization
        - target_col: the name of the target column for difference calculation
        - relative: whether to calculate the relative difference
        """
        if isinstance(index_col, (tuple, list)) and isinstance(target_col, (tuple, list)):
            assert len(index_col) == len(target_col) and len(index_col) == 2, "The length of two cols should be 2"
            rename_dict = {target_col[1]: target_col[0], index_col[1]: index_col[0]}
            if isinstance(ori_df, pd.DataFrame):
                df_1 = ori_df[[index_col[0], target_col[0]]].copy()
                df_2 = ori_df[[index_col[1], target_col[1]]].copy()
                df_1.set_index(index_col[0], inplace=True)
                df_2.set_index(index_col[1], inplace=True)
            elif isinstance(ori_df, Sequence):
                df_1 = ori_df[0][[index_col[0], target_col[0]]].copy()
                df_2 = ori_df[1][[index_col[1], target_col[1]]].copy()
                df_1.set_index(index_col[0], inplace=True)
                df_2.set_index(index_col[1], inplace=True)
            else:
                raise ValueError("check the type of ori_df and two cols")
            df_2.rename(columns=rename_dict, inplace=True)
        elif not isinstance(index_col, (tuple, list)) and not isinstance(target_col, (tuple, list)):
            assert isinstance(ori_df, Sequence), "check the type of ori_df and two cols"
            df_1 = ori_df[0][[index_col, target_col]].copy()
            df_2 = ori_df[1][[index_col, target_col]].copy()
            df_1.set_index(index_col, inplace=True)
            df_2.set_index(index_col, inplace=True)
        else:
            raise ValueError("two cols should be both list or both not list")

        common_idx = sorted(set(df_1.index).union(set(df_2.index)))
        df_1_reindexed = df_1.groupby(df_1.index).mean().reindex(common_idx).interpolate(method="linear").sort_index()
        df_2_reindexed = df_2.groupby(df_2.index).mean().reindex(common_idx).interpolate(method="linear").sort_index()
        diff = df_1_reindexed - df_2_reindexed
        if relative:
            diff = diff / df_2_reindexed
        diff[index_col[0]] = diff.index
        diff.reset_index(drop=True, inplace=True)
        return diff

    @staticmethod
    def identify_direction(ori_df: pd.DataFrame, idx_col: str | float | int, min_count: int = 17):
        """
        Identify the direction of the sweeping column and add another direction column
        (1 for increasing, -1 for decreasing)

        Args:
        - ori_df: the original dataframe
        - idx_col: the name of the index column
        - min_count: the min number of points for each direction (used to avoid fluctuation at ends)
        """
        df_in = ori_df.copy()
        df_in["direction"] = np.sign(np.gradient(df_in[idx_col]))
        directions = df_in['direction'].tolist()
        # Perform run-length encoding
        rle = [(direction, len(list(group))) for direction, group in groupby(directions)]
        # Initialize filtered directions list
        filtered_directions = []
        for idx, (direction, length) in enumerate(rle):
            if length >= min_count and direction != 0:
                # Accept the run as is
                filtered_directions.extend([direction] * length)
            else:
                # Replace short runs with the previous direction
                if filtered_directions:
                    replaced_direction = filtered_directions[-1]
                else:
                    lookahead_idx = idx + 1
                    while (lookahead_idx < len(rle) and
                           (rle[lookahead_idx][1] < min_count or rle[lookahead_idx][0] == 0)):
                        lookahead_idx += 1
                    assert lookahead_idx < len(rle), "The direction for starting is not clear"
                    replaced_direction = rle[lookahead_idx][0]
                filtered_directions.extend([replaced_direction] * length)

        # Assign the filtered directions back to the DataFrame
        df_in['direction'] = filtered_directions
        return df_in
