#!/usr/bin/env python
"""This module is responsible for processing and plotting the data"""

from __future__ import annotations
import importlib
from typing import List, Tuple
import matplotlib
import matplotlib.pyplot as plt
import copy
from common.file_organizer import FileOrganizer, script_base_dir
from common.measure_manager import MeasureManager
import common.pltconfig.color_preset as colors
from common.constants import cm_to_inch, factor, default_plot_dict
from common.data_process import DataProcess


class DataPlot(DataProcess):
    """
    This class is responsible for processing and plotting the data.
    Two series of functions will be provided, one is for automatic plotting, the other will provide dataframe or other data structures for manual plotting
    """
    # define static variables
    legend_font: dict
    """A constant dict used to set the font of the legend in the plot"""

    class PlotParam:
        """
        This class is used to store the parameters for the plot
        """
        def __init__(self, no_of_figs: int = 1) -> None:
            """
            initialize the PlotParam

            Args:
            - no_of_figs: the number of figures to be plotted
            """
            self.params_list = [copy.deepcopy(default_plot_dict) for _ in range(no_of_figs)]

        def set_param_dict(self, no_of_fig: int = 0, **kwargs) -> None:
            """
            set the parameters for the plot assignated by no_of_fig

            Args:
            - no_of_fig: the number of the figure to be set(start from 0)
            """
            self.params_list[no_of_fig].update(kwargs)

        def query_no(self) -> int:
            """
            query the number of figures to be plotted
            """
            return len(self.params_list)

        def merge(self, params2: 'PlotParam') -> None:
            """
            merge the parameters from another PlotParam to the end of the current one

            Args:
            - params: the PlotParam to be merged
            """
            self.params_list += params2.params_list


    def __init__(self, proj_name: str, *, no_params: int = 4, usetex: bool = False, usepgf: bool = False, if_folder_create = True) -> None:
        """
        Initialize the FileOrganizer and load the settings for matplotlib saved in another file
        
        Args:
        - proj_name: the name of the project
        - no_params: the number of params to be initiated (default:4) 
        - usetex: whether to use the TeX engine to render text
        - usepgf: whether to use the pgf backend to render text
        - if_folder_create: whether to create the folder for all the measurements in project
        """
        super().__init__(proj_name)
        DataPlot.load_settings(usetex, usepgf)
        self.create_folder("plot")
        self.unit = {"I":"A", "V":"V", "R":"Ohm", "T":"K", "B":"T","f":"Hz"}
        self.params = DataPlot.PlotParam(no_params)
        if if_folder_create:
            self.assign_folder()

    def assign_folder(self, folder_name:str = None) -> None:
        """ Assign the folder for the measurements """
        if folder_name is not None:
            self.create_folder(f"plot/{folder_name}")
        else:
            for i in self.query_proj()["measurements"]:
                self.create_folder(f"plot/{i}")

    @staticmethod
    def get_unit_factor_and_texname(unit: str) -> Tuple[float, str]:
        """
        Used in plotting, to get the factor and the TeX name of the unit
        
        Args:
        - unit: the unit name string (like: uA)
        """
        _factor = factor(unit)
        if unit[0] == "u":
            namestr = rf"$\mathrm{{\mu {unit[1:]}}}$".replace("Ohm",r"\Omega")
        else:
            namestr = rf"$\mathrm{{{unit}}}$".replace("Ohm",r"\Omega")
        return _factor, namestr

    def set_unit(self, unit_new:dict =None) -> None:
        """
        Set the unit for the plot, default to SI

        Args:
        - unit: the unit dictionary, the format is {"I":"uA", "V":"V", "R":"Ohm"}
        """
        self.unit.update(unit_new)

    def plot_RT(self, *, ax: matplotlib.axes.Axes = None, xylog = (False,False)) -> None:
        """
        plot the RT curve

        Args:
        - params: the PlotParam class containing the parameters for the plot, if None, the default parameters will be used.
        - ax: the axes to plot the figure
        - custom_unit: defined if the unit is not the default one(uA, V), the format is {"I":"uA", "V":"mV", "R":"mOhm"}
        """
        self.params.set_param_dict(0, label="RT")

        rt_df = self.dfs["RT"]
        if ax is None:
            if_indep = True
            fig, ax = DataPlot.init_canvas(2, 1, 10, 6)
        factor_r, unit_r_print = DataPlot.get_unit_factor_and_texname(self.unit["R"])
        factor_T, unit_T_print = DataPlot.get_unit_factor_and_texname(self.unit["T"])

        ax.plot(rt_df["T"]*factor_T, rt_df["R"]*factor_r, **self.params.params_list[0])
        ax.set_ylabel("$\\mathrm{R}$"+f"({unit_r_print})")
        ax.set_xlabel(f"$\\mathrm{{T}}$ ({unit_T_print})")
        ax.legend(edgecolor='black',prop=DataPlot.legend_font)
        if xylog[1]:
            ax.set_yscale("log")
        if xylog[0]:
            ax.set_xscale("log")

    def plot_nonlinear(self, *,
                       ax: matplotlib.axes.Axes = None,
                       ax2 : matplotlib.axes.Axes = None,
                       plot_order: Tuple[bool] = (True, True),
                       reverse_V: Tuple[bool] = (False, False),
                       in_ohm: bool = False,
                       xylog1 = (False,False), xylog2 = (False,False)) -> matplotlib.axes.Axes | None:
        """
        plot the nonlinear signals of a 1-2 omega measurement

        Parameters:
        params: PlotParam class containing the parameters for the 1st 
            signal plot, if None, the default parameters will be used.         
        params2: the PlotParam class for the 2nd harmonic signal
        plot_order : list of booleans
            [Vw, V2w]
        reverse_V : list of booleans
            if the voltage is reversed by adding a negative sign
        custom_unit : dict
            defined if the unit is not the default one(uA, V), the format is {"I":"uA", "V":"mV", "R":"mOhm"}
        """

        # assign and merge the plotting parameters
        self.params.set_param_dict(0, label="Vw")
        self.params.set_param_dict(1, label="V2w")

        nonlinear = self.dfs["nonlinear"]
        if_indep = False
        return_ax2 = False

        if reverse_V[0]:
            nonlinear["Vw"] = -nonlinear["Vw"]
        if reverse_V[1]:
            nonlinear["V2w"] = -nonlinear["V2w"]

        factor_i, unit_i_print = DataPlot.get_unit_factor_and_texname(self.unit["I"])
        factor_v, unit_v_print = DataPlot.get_unit_factor_and_texname(self.unit["V"])
        factor_r, unit_r_print = DataPlot.get_unit_factor_and_texname(self.unit["R"])

        if ax is None:
            if_indep = True
            fig, ax = DataPlot.init_canvas(1, 1, 10, 6)

        # plot the 2nd harmonic signal
        if plot_order[1]:
            if in_ohm:
                ax.plot(nonlinear["curr"]*factor_i, nonlinear["V2w"]*factor_r/nonlinear["curr"], **params.params_list[1])
                ax.set_ylabel("$\\mathrm{R^{2\\omega}}$"+f"({unit_r_print})")
            else:
                ax.plot(nonlinear["curr"]*factor_i, nonlinear["V2w"]*factor_v, **self.params.params_list[1])
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
                    ax2.plot(nonlinear["curr"]*factor_i, nonlinear["Vw"]*factor_r/nonlinear["curr"], **params.params_list[0])
                    ax2.set_ylabel("$\\mathrm{R^\\omega}$"+f"({unit_r_print})")
                else:
                    ax2.plot(nonlinear["curr"]*factor_i, nonlinear["Vw"]*factor_v, **params.params_list[0])
                    ax2.set_ylabel("$\\mathrm{V^\\omega}$"+f"({unit_v_print})")
                ax2.legend(edgecolor='black',prop=DataPlot.legend_font)
                if xylog1[1]:
                    ax2.set_yscale("log")
                if xylog1[0]:
                    pass
        else: # assume at least one is true
            if in_ohm:
                ax.plot(nonlinear["curr"]*factor_i, nonlinear["Vw"]*factor_r/nonlinear["curr"], **params.params_list[0])
                ax.set_ylabel("$\\mathrm{R^\\omega}$"+f"({unit_r_print})")
            else:
                ax.plot(nonlinear["curr"]*factor_i, nonlinear["Vw"]*factor_v, **params.params_list[0])
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
        file_name = "common.pltconfig.plot_config"
        if usetex:
            file_name += "_tex"
            if usepgf:
                file_name += "_pgf"
        else:
            file_name += "_notex"

        config_module = importlib.import_module(f"{file_name}", script_base_dir)
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

    def mapping(self):
        """
        This function is used to map the data to the corresponding functions
        """
        pass

    def live_plot(self, measure_name: str, *, ax: matplotlib.axes.Axes = None, **kwargs) -> None:
        """
        plot the live data

        Args:
        - measure_name: the name of the measurement
        - ax: the axes to plot the figure
        - kwargs: the parameters for the plot
        """
        ##TODO##
        pass