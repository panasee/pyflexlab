#!/usr/bin/env python
"""This module is responsible for processing and plotting the data"""

from __future__ import annotations
import importlib
from typing import List, Tuple, Literal
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import copy
from common.file_organizer import FileOrganizer, script_base_dir
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
        def __init__(self, *dims) -> None:
            """
            initialize the PlotParam

            Args:
            - no_of_figs: the number of figures to be plotted
            """
            self.shape = dims
            self.params_list = self._create_params_list(dims)
            # define a tmp params used for temporary storage, especially in class methods for convenience
            self.tmp = copy.deepcopy(default_plot_dict)
        
        def _create_params_list(self, dims: Tuple[int]) -> List[dict]:
            """
            create the list of parameters for the plot

            Args:
            - dims: the dimensions of the parameters
            """
            if len(dims) == 1:
                return [copy.deepcopy(default_plot_dict) for _ in range(dims[0])]
            else:
                return [self._create_params_list(dims[1:]) for _ in range(dims[0])]
        
        def _get_subarray(self, array, index: Tuple[int]) -> List[dict]:
            """
            get the subarray of the parameters for the plot assignated by the index
            """
            if len(index) == 1:
                return array[index[0]]
            else:
                return self._get_subarray(array[index[0]],index[1:])

        def _set_subarray(self, array, index: Tuple[int], target_dict:dict) -> None:
            """
            set the subarray of the parameters for the plot assignated by the index
            """
            if len(index) == 1:
                array[index[0]] = copy.deepcopy(target_dict)
            else:
                self._set_subarray(array[index[0]], index[1:], target_dict)
        
        def _flatten(self, lst):
            """
            Flatten a multi-dimensional list using recursion
            """
            return [item for sublist in lst for item in (self._flatten(sublist) if isinstance(sublist, list) else [sublist])]


        def __getitem__(self, index:Tuple[int] | int) -> dict:
            """
            get the parameters for the plot assignated by the index

            Args:
            - index: the index of the figure to be get
            """
            if isinstance(index, int):
                flat_list = self._flatten(self.params_list)
                return flat_list[index]
            result = self._get_subarray(self.params_list, index)
            while isinstance(result, list) and len(result) == 1:
                result = result[0]
            return result

        def __setitem__(self, index: Tuple[int] | int, value):
            if isinstance(index, int):
                index = (index,)
            self._set_subarray(self.params_list, index, value)

    def __init__(self, proj_name: str, *, no_params: Tuple[int] | int = 4, usetex: bool = False, usepgf: bool = False, if_folder_create = True) -> None:
        """
        Initialize the FileOrganizer and load the settings for matplotlib saved in another file
        
        Args:
        - proj_name: the name of the project
        - no_params: the number of params to be initiated (default:4) 
        - usetex: whether to use the TeX engine to render text
        - usepgf: whether to use the pgf backend
        - if_folder_create: whether to create the folder for all the measurements in project
        """
        super().__init__(proj_name)
        DataPlot.load_settings(usetex, usepgf)
        self.create_folder("plot")
        self.unit = {"I":"A", "V":"V", "R":"Ohm", "T":"K", "B":"T","f":"Hz"}
        # params here are mainly used for internal methods
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

    def df_plot_Tt(self, *, ax: matplotlib.axes.Axes = None) -> None:
        """
        Plot the T-t curve from the RT measurement (resolved to minute)
        """
        self.params.tmp.update(label="T-t")
        tt_df = self.dfs["RT"]["t1"] +" "+ self.dfs["RT"]["t2"]
        day_time = DataProcess.time_to_datetime(tt_df)
        ##TODO##

    def df_plot_RT(self, *, ax: matplotlib.axes.Axes = None, xylog = (False,False)) -> None:
        """
        plot the RT curve

        Args:
        - params: the PlotParam class containing the parameters for the plot, if None, the default parameters will be used.
        - ax: the axes to plot the figure
        - custom_unit: defined if the unit is not the default one(uA, V), the format is {"I":"uA", "V":"mV", "R":"mOhm"}
        """
        self.params.tmp.update(label="RT")

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

    def df_plot_nonlinear(self, *,
                       handlers: Tuple[matplotlib.axes.Axes] = None,
                       plot_order: Tuple[bool] = (True, True),
                       reverse_V: Tuple[bool] = (False, False),
                       in_ohm: bool = False,
                       xylog1 = (False,False), xylog2 = (False,False)) -> matplotlib.axes.Axes | None:
        """
        plot the nonlinear signals of a 1-2 omega measurement

        Args:
        handlers: Tuple[matplotlib.axes.Axes]
            the handlers for the plot, the content should be (ax_1w, ax_1w_phi,ax_2w, ax_2w)
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
        self.params[0].update(label=r"$V_w$")
        self.params[1].update(label=r"$V_{2w}$")
        self.params[2].update(label=r"$\phi_w$", color="c", linestyle="--",marker="",alpha=0.37)
        self.params[3].update(label=r"$\phi_{2w}$", color="m", linestyle="--",marker="",alpha=0.37)

        nonlinear = self.dfs["nonlinear"]
        if_indep = False
        return_handlers = False

        if reverse_V[0]:
            nonlinear["V1w"] = -nonlinear["V1w"]
        if reverse_V[1]:
            nonlinear["V2w"] = -nonlinear["V2w"]

        factor_i, unit_i_print = DataPlot.get_unit_factor_and_texname(self.unit["I"])
        factor_v, unit_v_print = DataPlot.get_unit_factor_and_texname(self.unit["V"])
        factor_r, unit_r_print = DataPlot.get_unit_factor_and_texname(self.unit["R"])

        if handlers is None:
            if_indep = True
            fig, ax = DataPlot.init_canvas(2, 1, 10, 12)
            ax_1w, ax_2w = ax
            ax_1w_phi = ax_1w.twinx()
            ax_2w_phi = ax_2w.twinx()
            return_handlers = True
        else:
            ax_1w, ax_1w_phi,ax_2w, ax_2w_phi = handlers

        # plot the 2nd harmonic signal
        if plot_order[1]:
            if in_ohm:
                line_v2w = ax_2w.plot(nonlinear["curr"]*factor_i, nonlinear["V2w"]*factor_r/nonlinear["curr"], **self.params.params_list[1])
                ax_2w.set_ylabel("$\\mathrm{R^{2\\omega}}$"+f"({unit_r_print})")
            else:
                line_v2w = ax_2w.plot(nonlinear["curr"]*factor_i, nonlinear["V2w"]*factor_v, **self.params.params_list[1])
                ax_2w.set_ylabel("$\\mathrm{V^{2\\omega}}$"+f"({unit_v_print})")
            
            line_v2w_phi = ax_2w_phi.plot(nonlinear["curr"]*factor_i, nonlinear["phi_2w"], **self.params.params_list[3])
            ax_2w_phi.set_ylabel(r"$\phi(\mathrm{^\circ})$")
            ax_2w.legend(handles = line_v2w+line_v2w_phi, labels = [line_v2w[0].get_label(), line_v2w_phi[0].get_label()], edgecolor='black',prop=DataPlot.legend_font)
            ax_2w.set_xlabel(f"I ({unit_i_print})")
            if xylog2[1]:
                ax_2w.set_yscale("log")
            if xylog2[0]:
                ax_2w.set_xscale("log")
            #ax.set_xlim(-0.00003,None)
        if plot_order[0]:
            if in_ohm:
                line_v1w = ax_1w.plot(nonlinear["curr"]*factor_i, nonlinear["V1w"]*factor_r/nonlinear["curr"], **self.params.params_list[0])
                ax_1w.set_ylabel("$\\mathrm{R^\\omega}$"+f"({unit_r_print})")
            else:
                line_v1w = ax_1w.plot(nonlinear["curr"]*factor_i, nonlinear["V1w"]*factor_v, **self.params.params_list[0])
                ax_1w.set_ylabel("$\\mathrm{V^\\omega}$"+f"({unit_v_print})")

            line_v1w_phi = ax_1w_phi.plot(nonlinear["curr"]*factor_i, nonlinear["phi_1w"], **self.params.params_list[3])
            ax_1w_phi.set_ylabel(r"$\phi(\mathrm{^\circ})$")
            ax_1w.legend(handles = line_v1w+line_v1w_phi, labels = [line_v1w[0].get_label(), line_v1w_phi[0].get_label()],edgecolor='black',prop=DataPlot.legend_font)
            if xylog1[1]:
                ax_1w.set_yscale("log")
            if xylog1[0]:
                ax_1w.set_xscale("log")
        if if_indep:
            fig.tight_layout()
            plt.show()
        if return_handlers:
            return (ax_1w, ax_1w_phi,ax_2w, ax_2w_phi)

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
    def paint_colors_twin_axes(*, ax_left:matplotlib.axes.Axes, color_left: str, ax_right:matplotlib.axes.Axes, color_right: str) -> None:
        """
        paint the colors for the twin y axes

        Args:
        - ax: the axes to paint the colors
        - left: the color for the left y axis
        - right: the color for the right y axis
        """
        ax_left.tick_params("y",colors=color_left)
        ax_left.spines["left"].set_color(color_left)
        ax_right.tick_params("y",colors=color_right)
        ax_right.spines["right"].set_color(color_right)


    @staticmethod
    def init_canvas(n_row: int, n_col: int, figsize_x: float, figsize_y: float, sub_adj: Tuple[float] = (0.19,0.13,0.97,0.97,0.2,0.2), **kwargs) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, DataPlot.PlotParam]:
        """
        initialize the canvas for the plot, return the fig and ax variables

        Args:
        - n_row: the fig no. of rows
        - n_col: the fig no. of columns
        - figsize_x: the width of the whole figure in cm
        - figsize_y: the height of the whole figure in cm
        - sub_adj: the adjustment of the subplots (left, bottom, right, top, wspace, hspace)
        - **kwargs: keyword arguments for the plt.subplots function
        """
        fig, ax = plt.subplots(n_row, n_col, figsize=(figsize_x * cm_to_inch, figsize_y * cm_to_inch), **kwargs)
        fig.subplots_adjust(left=sub_adj[0], bottom=sub_adj[1], right=sub_adj[2], top=sub_adj[3], wspace=sub_adj[4], hspace=sub_adj[5])
        return fig, ax, DataPlot.PlotParam(n_row,n_col,2)

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