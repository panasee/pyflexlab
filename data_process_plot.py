#!/usr/bin/env python

import importlib
import matplotlib
import matplotlib.pyplot as plt
from common.file_organizer import FileOrganizer
from common.measure_manager import MeasureManager
import pandas as pd
import numpy as np
import common.pltconfig.color_preset as colors
from common.constants import cm_to_inch, factor
from typing import List, Tuple


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

class DataPlot(DataProcess):
    """
    This class is responsible for processing and plotting the data.
    Two series of functions will be provided, one is for automatic plotting, the other will provide dataframe or other data structures for manual plotting
    """
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


    def plot_nonlinear(self, *, ax=None, ax2 =None, c1 =colors.Genshin["Nilou"][0], c2=colors.Genshin["Nilou"][0], lt1="-",lt2="-",l1 = "Vw", l2="V2w", plot_order=[True,True], reverse_V = [False,False], custom_unit={"I":"uA","V":"V","R":"Ohm"}, in_ohm=False, xylog1 = [False,False], xylog2 = [False,False]):
        """
        plot the nonlinear signals of a 1-2 omega measurement

        Parameters:
        plot_order : list of booleans
            [Vw, V2w]
        reverse_V : list of booleans
            if the voltage is reversed by adding a negative sign
        custom_unit : dict
            defined if the unit is not the default one(uA, V), the format is {"I":"uA", "V":"mV", "R":"mOhm"}
        """

        nhe = self.df

        if reverse_V[0]:
            nhe["Vw"] = -nhe["Vw"]
        if reverse_V[1]:
            nhe["V2w"] = -nhe["V2w"]

        if_indep = False
        return_ax2 = False

        unit_i = custom_unit["I"]
        unit_v = custom_unit["V"]
        unit_r = custom_unit["R"]

        factor_i = factor(unit_i)
        factor_v = factor(unit_v)
        factor_r = factor(unit_r)
        if unit_i == "uA":
            unit_i_print = "$\\mathrm{\\mu}$A"
        else:
            unit_i_print = unit_i
        if unit_v == "uV":
            unit_v_print = "$\\mathrm{\\mu}$V"
        else:
            unit_v_print = unit_v
        if unit_r == "uOhm":
            unit_r_print = "$\\mathrm{\\mu\\Omega}$"
        else:
            unit_r_print = unit_r.replace("Ohm","$\\mathrm{\\Omega}$")

        if ax is None:
            if_indep = True
            #fig, ax = plt.subplots(2,1,figsize=(6 * cm_to_inch, 10* cm_to_inch),sharex=True)
            fig, ax = plt.subplots(1,1,figsize=(10 * cm_to_inch,6.2 * cm_to_inch))
            fig.subplots_adjust(left=.19, bottom=.13, right=.97, top=.97)


        if plot_order[1]:
            if in_ohm:
                ax.plot(nhe["curr"]*factor_i, nhe["V2w"]*factor_r/nhe["curr"], lt2,color=c2,label=l2)
                ax.set_ylabel("$\\mathrm{R^{2\\omega}}$"+f"({unit_r_print})")
            else:
                ax.plot(nhe["curr"]*factor_i, nhe["V2w"]*factor_v, lt2,color=c2,label=l2)
                ax.set_ylabel("$\\mathrm{V^{2\\omega}}$"+f"({unit_v_print})")
            ax.legend(edgecolor='black',prop=DataPlot.legend_font)
            ax.set_xlabel(f"I ({unit_i_print})")
            if xylog2[1]:
                ax.set_yscale("log")
            if xylog2[0]:
                ax.set_xscale("log")
            #ax.set_xlim(-0.00003,None)
            if plot_order[0]:
                if ax2 is None:
                    ax2 = ax.twinx()
                    return_ax2 = True
                if in_ohm:
                    ax2.plot(nhe["curr"]*factor_i, nhe["Vw"]*factor_r/nhe["curr"], lt1,color=c1,label=l1)
                    ax2.set_ylabel("$\\mathrm{R^\\omega}$"+f"({unit_r_print})")
                else:
                    ax2.plot(nhe["curr"]*factor_i, nhe["Vw"]*factor_v, lt1,color=c1,label=l1)
                    ax2.set_ylabel("$\\mathrm{V^\\omega}$"+f"({unit_v_print})")
                ax2.legend(edgecolor='black',prop=DataPlot.legend_font)
                if xylog1[1]:
                    ax2.set_yscale("log")
                if xylog1[0]:
                    pass
        else: # assume at least one is true
            if in_ohm:
                ax.plot(nhe["curr"]*factor_i, nhe["Vw"]*factor_r/nhe["curr"], lt1,color=c1,label=l1)
                ax.set_ylabel("$\\mathrm{R^\\omega}$"+f"({unit_r_print})")
            else:
                ax.plot(nhe["curr"]*factor_i, nhe["Vw"]*factor_v, lt1,color=c1,label=l1)
                ax.set_ylabel("$\\mathrm{V^\\omega}$"+f"({unit_v_print})")
            ax.legend(edgecolor='black',prop=DataPlot.legend_font)
            ax.set_xlabel(f"I ({unit_i_print})")
            if xylog1[1]:
                ax.set_yscale("log")
            if xylog1[0]:
                ax.set_yscale("log")


        if if_indep:
            plt.show()
        if return_ax2:
            return ax2

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

    @staticmethod
    def init_canvas(n_row: int, n_col: int, figsize_x: float, figsize_y: float) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """
        initialize the canvas for the plot, return the fig and ax variables

        Args:
        - n_row: the fig no. of rows
        - n_col: the fig no. of columns
        - figsize_x: the width of the whole figure in cm
        - figsize_y: the height of the whole figure in cm
        """
        fig, ax = plt.subplots(n_row, n_col, figsize=(figsize_x * cm_to_inch, figsize_y * cm_to_inch))
        fig.subplots_adjust(left=.19, bottom=.13, right=.97, top=.97)
        return fig, ax

