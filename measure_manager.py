#!/usr/bin/env python

"""This module is responsible for managing the measure-related folders and data Note each instrument better be
initialzed right before the measurement, as there may be a long time between loading and measuremnt, leading to
possibilities of parameter changing"""
from typing import List, Tuple, Literal
import time
import datetime
from abc import ABC, abstractmethod
import numpy as np
from pymeasure.instruments.srs import SR830
from pymeasure.instruments.oxfordinstruments import ITC503
from pymeasure.instruments.keithley import Keithley6221
from pymeasure.instruments.keithley import Keithley2182
import pyvisa
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
# import qcodes as qc
from qcodes.instrument_drivers.oxford import OxfordMercuryiPS
from qcodes.instrument_drivers.Keithley import Keithley2400

from common.file_organizer import print_help_if_needed, FileOrganizer
from common.data_plot import DataPlot
from common.constants import factor
from common.mercuryITC import MercuryITC
from common.Keithley_6430 import Keithley_6430


class MeasureManager(DataPlot):
    """This class is a subclass of FileOrganizer and is responsible for managing the measure-related folders and data"""

    def __init__(self, proj_name: str) -> None:
        """Note that the FileOrganizer.out_database_init method should be called to assign the correct path to the
        out_database attribute. This method should be called before the MeasureManager object is created."""
        super().__init__(proj_name)  # Call the constructor of the parent class
        self.instrs = {}
        # load params for plotting in measurement
        DataPlot.load_settings(False, False)

    def load_SR830(self, *addresses: Tuple[str]) -> None:
        """
        load SR830 instruments according the addresses, store them in self.instrs["sr830"] in corresponding order

        Args:
            addresses (List[str]): the addresses of the SR830 instruments (take care of the order)
        """
        self.instrs["sr830"] = []
        for addr in addresses:
            self.instrs["sr830"].append(SR830(addr))
        self.setup_SR830()

    def load_2182(self, address: str) -> None:
        """
        load Keithley 2182 instrument according to the address, store it in self.instrs["2182"]
        """
        self.instrs["2182"] = Keithley2182(address)

    def load_2400(self, address: str) -> None:
        """
        load Keithley 2400 instrument according to the address, store it in self.instrs["2400"]
        """
        self.instrs["2400"] = Keithley2400("2400", address)

    def load_6221(self, address: str) -> None:
        """
        load Keithley 6221 instrument according to the address, store it in self.instrs["6221"]
        """
        self.instrs["6221"] = Keithley6221(address)
        self.setup_6221()

    def load_6430(self, address: str) -> None:
        """
        load Keithley 6430 instrument according to the address, store it in self.instrs["6430"]
        """
        self.instrs["6430"] = Keithley_6430("SMU", address)

    def load_ITC503(self, gpib_up: str, gpib_down: str) -> None:
        """
        load ITC503 instruments according to the addresses, store them in self.instrs["itc503"] in corresponding order. Also store the ITC503 instruments in self.instrs["itc"] for convenience to call

        Args:
            addresses (List[str]): the addresses of the ITC503 instruments (take care of the order)
        """
        self.instrs["itc503"] = ITCs(gpib_up, gpib_down)
        self.instrs["itc"] = self.instrs["itc503"]

    def load_mercury_ips(self, address: str = "TCPIP0::10.97.24.237::7020::SOCKET", if_print: bool = False,
                         limit_sphere: float = 14) -> None:
        """
        load Mercury iPS instrument according to the address, store it in self.instrs["mercury_ips"]

        Args:
            address (str): the address of the instrument
            if_print (bool): whether to print the snapshot of the instrument
            limit_sphere (float): the limit of the field
        """
        self.instrs["mercury_ips"] = OxfordMercuryiPS("mips", address)
        if if_print:
            self.instrs["mercury_ips"].print_readable_snapshot(update=True)

        def spherical_limit(x, y, z) -> bool:
            return np.sqrt(x ** 2 + y ** 2 + z ** 2) <= limit_sphere

        self.instrs["mercury_ips"].set_new_field_limits(spherical_limit)

    def ramp_magfield(self, field: float | Tuple[float], rate: Tuple[float] = (0.00333,) * 3, wait: bool = True,
                      tolerance: float = 3e-3, if_plot: bool = False) -> None:
        """
        ramp the magnetic field to the target value with the rate, current the field is only in Z direction limited by the actual instrument setting

        Args:
            field (Tuple[float]): the target field coor
            rate (float): the rate of the field change (T/s)
            wait (bool): whether to wait for the ramping to finish
            tolerance (float): the tolerance of the field (T)
        """
        mips = self.instrs["mercury_ips"]
        if max(rate) * 60 > 0.2:
            raise ValueError("The rate is too high, the maximum rate is 0.2 T/min")
        #mips.GRPX.field_ramp_rate(rate[0])
        #mips.GRPY.field_ramp_rate(rate[1])
        mips.GRPZ.field_ramp_rate(rate[2])
        # no x and y field for now
        #mips.x_target(field[0])
        #mips.y_target(field[1])
        if isinstance(field, (tuple, list)):
            mips.z_target(field[2])
        if isinstance(field, (float, int)):
            mips.z_target(field)

        mips.ramp(mode="simul")
        if wait:
            # the is_ramping() method is not working properly, so we use the following method to wait for the ramping
            # to finish
            if if_plot:
                self.live_plot_init(1, 1, 1, 600, 1400, titles=[["H ramping"]],
                                    axes_labels=[[[r"Time (s)", r"Field_norm (T)"]]])
            time_arr = [0]
            field_arr = [np.linalg.norm((mips.x_measured(), mips.y_measured(), mips.z_measured()))]
            count = 0
            step_count = 1
            stability_counter = 20  # [s]
            while count < stability_counter:
                field_now = (mips.x_measured(), mips.y_measured(), mips.z_measured())
                time_arr.append(step_count)
                field_arr.append(np.linalg.norm(field_now))
                if abs(np.linalg.norm(field_now) - np.linalg.norm(field)) < tolerance:
                    count += 1
                else:
                    count = 0
                MeasureManager.print_progress_bar(count, stability_counter, prefix="Stablizing",
                                                  suffix=f"B: {field_now} T")
                time.sleep(1)
                if if_plot:
                    self.live_plot_update(0, 0, 0, time_arr, field_arr)
                step_count += 1
            print("ramping finished")

    def load_mercury_itc(self, address: str = "TCPIP0::10.101.28.24::7020::SOCKET") -> None:
        """
        load Mercury iPS instrument according to the address, store it in self.instrs["mercury_ips"]
        """
        #self.instrs["mercury_itc"] = MercuryITC(address)
        self.instrs["mercury_itc"] = ITCMercury(address)
        self.instrs["itc"] = self.instrs["mercury_itc"]
        #print(self.instrs["mercury_itc"].modules)

    def setup_SR830(self) -> None:
        """
        setup the SR830 instruments using pre-stored setups here, this function will not fully reset the instruments,
        only overwrite the specific settings here, other settings will all be reserved
        """
        for instr in self.instrs["sr830"]:
            instr.filter_slope = 24
            instr.time_constant = 0.3
            instr.input_config = "A - B"
            instr.input_coupling = "AC"
            instr.input_grounding = "Float"
            instr.sine_voltage = 0
            instr.input_notch_config = "None"
            instr.reference_source = "External"
            instr.reserve = "High Reserve"
            instr.filter_synchronous = False

    def setup_2182(self, channel: Literal[0, 1, 2] = 1) -> None:
        """
        set up the Keithley 2182 instruments, overwrite the specific settings here, other settings will all be reserved
        currently initialized to measure voltage
        """
        source_2182 = self.instrs["2182"]
        source_2182.reset()
        source_2182.active_channel = channel
        source_2182.channel_function = "voltage"
        source_2182.voltage_nplc = 5
        #source_2182.sample_continuously()
        #source_2182.ch_1.voltage_offset_enabled = True
        #source_2182.ch_1.acquire_voltage_reference()
        source_2182.ch_1.setup_voltage()

    def setup_6221(self, mode: Literal["ac", "dc"] = "ac", *, offset=0) -> None:
        """
        set up the Keithley 6221 instruments, overwrite the specific settings here, other settings will all be reserved. Note that the waveform will not begin here
        """
        source_6221 = self.instrs["6221"]
        source_6221.clear()
        if mode == "ac":
            source_6221.waveform_function = "sine"
            source_6221.waveform_amplitude = 0
            source_6221.waveform_offset = offset
            source_6221.waveform_ranging = "best"
            source_6221.waveform_use_phasemarker = True
            source_6221.waveform_phasemarker_line = 3
            source_6221.waveform_duration_set_infinity()
            source_6221.waveform_phasemarker_phase = 0
        source_6221.source_auto_range = False
        source_6221.output_low_grounded = False

    def measure_contact_6430(self, v_max: float = 1E-4, v_step: float = 1E-5, curr_compliance: float = 1E-6,
                             mode: Literal["0-max-0", "0--max-max-0"] = "0-max-0", *, test: bool = True,
                             source: int = None, drain: int = None, temp: int = None, tmpfolder: str = None) -> None:
        """
        Measure the IV curve using Keithley 6430 to test the contacts. No file will be saved if test is True
        """
        print(f"Max Voltage: {v_max} V")
        print(f"Voltage Step: {v_step} V")
        print(f"Max Curr: {curr_compliance} A")
        print(f"Meter: {self.instrs['6430']}")
        measure_delay = 0.3  # [s]
        instr_6430 = self.instrs["6430"]
        instr_6430.sense_mode("CURR:DC")
        instr_6430.source_mode("VOLT")
        if v_max > 200:
            raise ValueError("The maximum voltage is too high")
        elif v_max < 0.21 * 0.7:
            instr_6430.source_voltage_range(0.21)
        elif v_max > 140:
            instr_6430.source_voltage_range(200)
        else:
            instr_6430.source_voltage_range(v_max / 0.7)
        instr_6430.source_current_compliance(curr_compliance)
        instr_6430.output_enabled(True)
        #tmp_df = pd.DataFrame(columns=["V","V_sensed", "I", "R"])
        tmp_df = pd.DataFrame(columns=["V", "I"])
        self.live_plot_init(1, 1, 1, 600, 1400, titles=[["IV 6430"]], axes_labels=[[[r"Voltage (V)", r"Current (A)"]]])

        v_array = np.arange(0, v_max, v_step)
        if mode == "0-max-0":
            v_array = np.concatenate((v_array, v_array[::-1]))
        elif mode == "0--max-max-0":
            v_array = np.concatenate((-v_array, -v_array[::-1], v_array, v_array[::-1]))
        tmp_df["V"] = v_array
        i_array = np.zeros_like(v_array)
        v_sensed_array = np.zeros_like(v_array)
        r_array = np.zeros_like(v_array)

        for ii, v in enumerate(v_array):
            instr_6430.source_voltage(v)
            time.sleep(measure_delay)
            #i_array[ii] = instr_6430.ask(":READ?")
            i_array[ii] = instr_6430.sense_current()
            #v_sensed_array[ii] = instr_6430.sense_voltage()
            #r_array[ii] = instr_6430.sense_resistance()
            self.live_plot_update(0, 0, 0, tmp_df["V"][:ii + 1], i_array[:ii + 1])

        tmp_df["I"] = i_array
        #tmp_df["V_sensed"] = v_sensed_array
        #tmp_df["R"] = r_array
        if not test:
            file_path = self.get_filepath("IV__2-terminal", v_max, v_step, source, drain, mode, temp,
                                          tmpfolder=tmpfolder)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_df.to_csv(file_path, sep="\t", index=False, float_format="%.12f")

        instr_6430.output_enabled(False)

    def measure_contact_2400(self, v_max: float = 1E-4, v_step: float = 1E-5, curr_compliance: float = 1E-6,
                             mode: Literal["0-max-0", "0--max-max-0"] = "0-max-0", *, test: bool = True,
                             source: int = None,
                             drain: int = None, temp: int = None, tmpfolder: str = None) -> None:
        """
        Measure the IV curve using Keithley 2400 to test the contacts. No file will be saved if test is True

        Args:
            v_max (float): the maximum voltage to apply
            v_step (float): the step of the voltage
            curr_compliance (float): the current compliance
            mode (Literal["0-max-0","0--max-max-0"]): the mode of the measurement
            test (bool): whether for testing use, if True, no data will be saved, and the parameters after will not be used
        """
        print(f"Max Voltage: {v_max} V")
        print(f"Voltage Step: {v_step} V")
        print(f"Max Curr: {curr_compliance} A")
        print(f"Meter: {self.instrs['2400'].adapter}")
        measure_delay = 0.3  # [s]
        instr_2400 = self.instrs["2400"]
        instr_2400.compliancei(curr_compliance)
        #instr_2400.apply_voltage(compliance_current=curr_compliance)
        #instr_2400.source_voltage = 0
        #instr_2400.enable_source()
        #instr_2400.measure_current()
        tmp_df = pd.DataFrame(columns=["V", "I"])
        self.live_plot_init(1, 1, 1, 600, 1600, titles=[["IV 2400"]], axes_labels=[[[r"Voltage (V)", r"Current (A)"]]])

        v_array = np.arange(0, v_max, v_step)
        if mode == "0-max-0":
            v_array = np.concatenate((v_array, v_array[::-1]))
        elif mode == "0--max-max-0":
            v_array = np.concatenate((-v_array, -v_array[::-1], v_array, v_array[::-1]))
        tmp_df["V"] = v_array
        i_array = np.zeros_like(v_array)

        for ii, v in enumerate(v_array):
            instr_2400.ramp_to_voltage(v, steps=10, pause=0.03)
            time.sleep(measure_delay)
            i_array[ii] = instr_2400.current
            self.live_plot_update(0, 0, 0, tmp_df["V"][:ii + 1], i_array[:ii + 1])
        tmp_df["I"] = i_array
        if not test:
            file_path = self.get_filepath("IV__2-terminal", v_max, v_step, source, drain, mode, temp,
                                          tmpfolder=tmpfolder)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_df.to_csv(file_path, sep="\t", index=False, float_format="%.12f")

        instr_2400.shutdown()

    @print_help_if_needed
    def measure_VB_SR830(self, measurename_all, *var_tuple, source: Literal["sr830", "6221"],
                         resistor: float = None) -> None:
        """
        measure voltage signal of constant current under different temperature (continuously changing).

        Args:
            measurename_all (str): the full name of the measurement
            var_tuple (Tuple): the variables of the measurement, use "-h" to see the available options
            source (Literal["sr830","6221"]): the source of the measurement
            resistor (float): the resistance of the resistor, used only for sr830 source to calculate the voltage
        """
        file_path = self.get_filepath(measurename_all, *var_tuple)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.add_measurement(measurename_all)
        sub_type = FileOrganizer.measurename_decom(measurename_all)[-1]
        fast = False
        if var_tuple[2] < 2:
            raise ValueError("npts should be no less than 2")
        elif var_tuple[2] == 2:
            fast = True
        curr = MeasureManager.split_no_str(var_tuple[3])
        curr = factor(curr[1], "to_SI") * curr[0]  # [A]
        print(f"Filename is: {file_path.name}")
        print(f"Curr: {curr} A")
        print(f"Mag: {var_tuple[0]} -> {var_tuple[1]} T")
        print(f"steps: {var_tuple[2] - 1}")
        print(f"fast mode: {fast}")
        # for two types of measurements, the V1 meter is the same as the 2w meter,
        # and the V2 meter is the same as the 1w meter
        # so the variable name will not be changed for 2pair measurement
        self.print_pairs(sub_type, 0, 1)
        B_arr = np.linspace(var_tuple[0], var_tuple[1], var_tuple[2])
        freq = var_tuple[4]
        tmp_df = pd.DataFrame(columns=["B", "X_2w", "Y_2w", "R_2w", "phi_2w", "X_1w", "Y_1w", "R_1w", "phi_1w", "T"])
        out_range = False

        self.setup_SR830()
        meter_2w = self.instrs['sr830'][0]
        meter_1w = self.instrs['sr830'][1]
        meter_2w.harmonic = 2
        meter_1w.harmonic = 1

        if "1pair" in sub_type.split("-"):
            meter_2w.harmonic = 2
            meter_1w.harmonic = 1
            self.live_plot_init(3, 2, 2, 600, 1400,
                                titles=[["2w", "phi"], ["1w", "phi"], ["T", ""]],
                                axes_labels=[[["B (T)", "V2w (V)"], ["B (T)", "phi"]],
                                             [["B (T)", "V1w (V)"], ["B (T)", "phi"]],
                                             [["t", "B (T)"], ["", ""]]],
                                line_labels=[[["X", "Y"], ["", ""]],
                                             [["X", "Y"], ["", ""]],
                                             [["", ""], ["", ""]]])
        elif "2pair" in sub_type.split("-"):
            self.live_plot_init(3, 2, 2, 600, 1400,
                                titles=[["V1", "phi"], ["V1", "phi"], ["T", ""]],
                                axes_labels=[[["B (T)", "V1 (V)"], ["B (T)", "phi"]],
                                             [["B (T)", "V2 (V)"], ["B (T)", "phi"]],
                                             [["t", "B (T)"], ["", ""]]],
                                line_labels=[[["X", "Y"], ["", ""]],
                                             [["X", "Y"], ["", ""]],
                                             [["", ""], ["", ""]]])
        if source == "sr830":
            meter_1w.reference_source_trigger = "POS EDGE"
            meter_2w.reference_source_trigger = "SINE"
            meter_2w.reference_source = "Internal"
            meter_2w.frequency = freq
            if resistor is None:
                raise ValueError("resistor is needed for sr830 source")
        elif source == "6221":
            # 6221 use half peak-to-peak voltage as amplitude
            curr_p2p = curr * np.sqrt(2)
            source_6221 = self.instrs["6221"]
            self.setup_6221()
            source_6221.source_compliance = curr_p2p * 1000  # compliance voltage
            source_6221.source_range = curr_p2p / 0.6
            print(f"Keithley 6221 source range is set to {curr_p2p / 0.6} A")
            source_6221.waveform_frequency = freq
            meter_1w.reference_source_trigger = "POS EDGE"
            meter_2w.reference_source_trigger = "POS EDGE"
        try:
            if source == "sr830":
                for i in np.arange(0, curr * resistor, 0.02):
                    meter_2w.sine_voltage = i
                    time.sleep(0.5)
                meter_2w.sine_voltage = curr * resistor
            elif source == "6221":
                source_6221.waveform_abort()
                source_6221.waveform_amplitude = curr_p2p
                source_6221.waveform_arm()
                source_6221.waveform_start()

            time_arr = []
            if not fast:
                for i, b_i in enumerate(B_arr):
                    self.ramp_magfield(b_i, wait=True)
                    if meter_1w.is_out_of_range():
                        out_range = True
                    elif meter_2w.is_out_of_range():
                        out_range = True
                    list_2w = meter_2w.snap("X", "Y", "R", "THETA")
                    list_1w = meter_1w.snap("X", "Y", "R", "THETA")
                    temp = self.instrs["itc"].temperature
                    list_tot = [b_i] + list_2w + list_1w + [temp]
                    print(f"B: {list_tot[0]:.4f} T\t 2w: {list_tot[1:5]}\t 1w: {list_tot[5:9]}\t T: {list_tot[-1]}")
                    tmp_df.loc[len(tmp_df)] = list_tot
                    time_arr.append(datetime.datetime.now())
                    self.live_plot_update([0, 0, 0, 1, 1, 1, 2],
                                          [0, 0, 1, 0, 0, 1, 0],
                                          [0, 1, 0, 0, 1, 0, 0],
                                          [tmp_df["B"]] * 6 + [time_arr],
                                          np.array(tmp_df[["X_2w", "Y_2w", "phi_2w", "X_1w", "Y_1w", "phi_1w", "T"]]).T)
                    if i % 3 == 0:
                        tmp_df.to_csv(file_path, sep="\t", index=False, float_format="%.12f")
            if fast:
                i = 0
                counter = 0
                #self.ramp_magfield(B_arr[0], wait=True)
                self.ramp_magfield(B_arr[1], wait=False)
                while counter < 300:
                    i += 1
                    if meter_1w.is_out_of_range():
                        out_range = True
                    elif meter_2w.is_out_of_range():
                        out_range = True
                    list_2w = meter_2w.snap("X", "Y", "R", "THETA")
                    list_1w = meter_1w.snap("X", "Y", "R", "THETA")
                    temp = self.instrs["itc"].temperature
                    B_now_z = self.instrs["mercury_ips"].z_measured()
                    list_tot = [B_now_z] + list_2w + list_1w + [temp]
                    time_arr.append(datetime.datetime.now())
                    print(f"B: {list_tot[0]:.4f} T\t 2w: {list_tot[1:5]}\t 1w: {list_tot[5:9]}\t T: {list_tot[-1]}")
                    tmp_df.loc[len(tmp_df)] = list_tot
                    self.live_plot_update([0, 0, 0, 1, 1, 1, 2],
                                          [0, 0, 1, 0, 0, 1, 0],
                                          [0, 1, 0, 0, 1, 0, 0],
                                          [tmp_df["B"]] * 6 + [time_arr],
                                          np.array(tmp_df[["X_2w", "Y_2w", "phi_2w", "X_1w", "Y_1w", "phi_1w", "T"]]).T)
                    if i % 7 == 0:
                        tmp_df.to_csv(file_path, sep="\t", index=False, float_format="%.12f")
                    time.sleep(1)

                    if abs(self.instrs["mercury_ips"].z_measured() - var_tuple[1]) < 0.003:
                        counter += 1
                    else:
                        counter = 0
            self.dfs["VB"] = tmp_df.copy()
            # rename the columns for compatibility with the plotting function
            self.set_unit({"I": "uA", "V": "uV"})
            #self.df_plot_nonlinear(handlers=(ax[1],phi[1],ax[0],phi[0]))
            if out_range:
                print("out-range happened, rerun")
        except KeyboardInterrupt:
            print("Measurement interrupted")
        finally:
            tmp_df.to_csv(file_path, sep="\t", index=False, float_format="%.12f")
            if source == "sr830":
                meter_2w.sine_voltage = 0
            if source == "6221":
                source_6221.shutdown()

    @print_help_if_needed
    def measure_VT_SR830(self, measurename_all, *var_tuple, source: Literal["sr830", "6221"], resistor: float = None,
                         stability_counter: int = 120, thermalize_counter: int = 120, ramp_rate: float = 5) -> None:
        """
        measure voltage signal of constant current under different temperature (continuously changing).
        NOTE: set npts to 2 to do fast ramping.
        The normal ramping record data at every temperature point after it's been stablized and thermalized, while the fast ramping record according to time on the way

        Args:
            measurename_all (str): the full name of the measurement
            var_tuple (Tuple): the variables of the measurement, use "-h" to see the available options
            source (Literal["sr830","6221"]): the source of the measurement
            resistor (float): the resistance of the resistor, used only for sr830 source to calculate the voltage
            stability_counter (int, [s]): the counter for the stability of the temperature
            thermalize_counter (int, [s]): the counter for the thermalization of the temperature
            ramp_rate (float, [K/min]): the rate of the temperature ramping
        """
        sub_type = FileOrganizer.measurename_decom(measurename_all)[-1]
        fast = False
        if var_tuple[2] < 2:
            raise ValueError("npts should be no less than 2")
        elif var_tuple[2] == 2:
            fast = True
        file_path = self.get_filepath(measurename_all, *var_tuple)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.add_measurement(measurename_all)
        curr = MeasureManager.split_no_str(var_tuple[3])
        curr = factor(curr[1], "to_SI") * curr[0]  # [A]
        print(f"Filename is: {file_path.name}")
        print(f"Curr: {curr} A")
        print(f"Temperature: {var_tuple[0]} -> {var_tuple[1]} K")
        print(f"steps: {var_tuple[2] - 1}")
        print(f"fast mode: {fast}")
        # for two types of measurements, the V1 meter is the same as the 2w meter,
        # and the V2 meter is the same as the 1w meter
        # so the variable name will not be changed for 2pair measurement
        if "1pair" in sub_type.split("-"):
            print(f"2w meter: {self.instrs['sr830'][0].adapter}")
            print(f"1w meter: {self.instrs['sr830'][1].adapter}")
        elif "2pair" in sub_type.split("-"):
            print("===========================================")
            print(f"V1 meter: {self.instrs['sr830'][0].adapter}\t ORDER: {self.instrs['sr830'][0].harmonic}")
            print(f"V2 meter: {self.instrs['sr830'][1].adapter}\t ORDER: {self.instrs['sr830'][1].harmonic}")
            print("===========================================")
        T_arr = np.linspace(var_tuple[0], var_tuple[1], var_tuple[2])
        freq = var_tuple[4]
        tmp_df = pd.DataFrame(columns=["T", "X_2w", "Y_2w", "R_2w", "phi_2w", "X_1w", "Y_1w", "R_1w", "phi_1w", "curr"])
        out_range = False

        self.setup_SR830()
        meter_2w = self.instrs['sr830'][0]
        meter_1w = self.instrs['sr830'][1]
        if "1pair" in sub_type.split("-"):
            meter_2w.harmonic = 2
            meter_1w.harmonic = 1
            self.live_plot_init(3, 2, 2, 600, 1400,
                                titles=[["2w", "phi"], ["1w", "phi"], ["T", ""]],
                                axes_labels=[[["T (K)", "V2w (V)"], ["T (K)", "phi"]],
                                             [["T (K)", "V1w (V)"], ["T (K)", "phi"]],
                                             [["t(min)", "T (K)"], ["", ""]]],
                                line_labels=[[["X", "Y"], ["", ""]],
                                             [["X", "Y"], ["", ""]],
                                             [["", ""], ["", ""]]])
        elif "2pair" in sub_type.split("-"):
            self.live_plot_init(3, 2, 2, 600, 1400,
                                titles=[["V1", "phi"], ["V1", "phi"], ["T", ""]],
                                axes_labels=[[["T (K)", "V1 (V)"], ["T (K)", "phi"]],
                                             [["T (K)", "V2 (V)"], ["T (K)", "phi"]],
                                             [["t(min)", "T (K)"], ["", ""]]],
                                line_labels=[[["X", "Y"], ["", ""]],
                                             [["X", "Y"], ["", ""]],
                                             [["", ""], ["", ""]]])
        if source == "sr830":
            meter_1w.reference_source_trigger = "POS EDGE"
            meter_2w.reference_source_trigger = "SINE"
            meter_2w.reference_source = "Internal"
            meter_2w.frequency = freq
            if resistor is None:
                raise ValueError("resistor is needed for sr830 source")
        elif source == "6221":
            # 6221 use half peak-to-peak voltage as amplitude
            curr_p2p = curr * np.sqrt(2)
            source_6221 = self.instrs["6221"]
            self.setup_6221()
            source_6221.source_compliance = curr_p2p * 10000  # compliance voltage
            source_6221.source_range = curr_p2p / 0.6
            print(f"Keithley 6221 source range is set to {curr_p2p / 0.6} A")
            source_6221.waveform_frequency = freq
            meter_1w.reference_source_trigger = "POS EDGE"
            meter_2w.reference_source_trigger = "POS EDGE"
        try:
            if source == "sr830":
                for i in np.arange(0, curr * resistor, 0.02):
                    meter_2w.sine_voltage = i
                    time.sleep(0.39)
                meter_2w.sine_voltage = curr * resistor
            elif source == "6221":
                source_6221.waveform_abort()
                source_6221.waveform_amplitude = curr_p2p
                source_6221.waveform_arm()
                source_6221.waveform_start()

            time_arr = []
            if not fast:
                for i, temp_i in enumerate(T_arr):
                    self.instrs["itc"].ramp_to_temperature(temp_i, ramp_rate=ramp_rate,
                                                           stability_counter=stability_counter,
                                                           thermalize_counter=thermalize_counter)
                    if meter_1w.is_out_of_range():
                        out_range = True
                    elif meter_2w.is_out_of_range():
                        out_range = True
                    list_2w = meter_2w.snap("X", "Y", "R", "THETA")
                    list_1w = meter_1w.snap("X", "Y", "R", "THETA")
                    temp = self.instrs["itc"].temperature
                    list_tot = [temp] + list_2w + list_1w + [curr]
                    if "1pair" in sub_type.split("-"):
                        print(f"T: {list_tot[0]:.2f} K\t 2w: {list_tot[1:5]}\t 1w: {list_tot[5:9]}")
                    elif sub_type.split("-")[-1] == "2pair":
                        print(f"T: {list_tot[0]:.2f} K\t V1: {list_tot[1:5]}\t V2: {list_tot[5:9]}")
                    tmp_df.loc[len(tmp_df)] = list_tot
                    time_arr.append(datetime.datetime.now())
                    self.live_plot_update([0, 0, 0, 1, 1, 1, 2],
                                          [0, 0, 1, 0, 0, 1, 0],
                                          [0, 1, 0, 0, 1, 0, 0],
                                          [tmp_df["T"]] * 6 + [time_arr],
                                          np.array(tmp_df[["X_2w", "Y_2w", "phi_2w", "X_1w", "Y_1w", "phi_1w", "T"]]).T)
                    if i % 3 == 0:
                        tmp_df.to_csv(file_path, sep="\t", index=False, float_format="%.12f")
            if fast:
                i = 0
                counter = 0
                self.instrs["itc"].ramp_to_temperature(var_tuple[0], ramp_rate=ramp_rate, wait=True,
                                                       stability_counter=stability_counter,
                                                       thermalize_counter=thermalize_counter)
                self.instrs["itc"].ramp_to_temperature(var_tuple[1], ramp_rate=ramp_rate, wait=False,
                                                       stability_counter=stability_counter,
                                                       thermalize_counter=thermalize_counter)
                # assume 600s to end the RT curve
                while counter < 600:
                    i += 1
                    if meter_1w.is_out_of_range():
                        out_range = True
                    elif meter_2w.is_out_of_range():
                        out_range = True
                    list_2w = meter_2w.snap("X", "Y", "R", "THETA")
                    list_1w = meter_1w.snap("X", "Y", "R", "THETA")
                    temp = self.instrs["itc"].temperature
                    list_tot = [temp] + list_2w + list_1w + [curr]
                    if "1pair" in sub_type.split("-"):
                        print(f"T: {list_tot[0]:.2f} K\t 2w: {list_tot[1:5]}\t 1w: {list_tot[5:9]}")
                    elif "2pair" in sub_type.split("-"):
                        print(f"T: {list_tot[0]:.2f} K\t V1: {list_tot[1:5]}\t V2: {list_tot[5:9]}")
                    tmp_df.loc[len(tmp_df)] = list_tot
                    time_arr.append(datetime.datetime.now())
                    self.live_plot_update([0, 0, 0, 1, 1, 1, 2],
                                          [0, 0, 1, 0, 0, 1, 0],
                                          [0, 1, 0, 0, 1, 0, 0],
                                          [tmp_df["T"]] * 6 + [time_arr],
                                          np.array(tmp_df[["X_2w", "Y_2w", "phi_2w", "X_1w", "Y_1w", "phi_1w", "T"]]).T)
                    if i % 7 == 0:
                        tmp_df.to_csv(file_path, sep="\t", index=False, float_format="%.12f")
                    time.sleep(1)

                    if abs(temp - var_tuple[1]) < ITC.dynamic_delta(var_tuple[1], 0.01):
                        counter += 1
                    else:
                        counter = 0
            self.dfs["VT"] = tmp_df.copy()
            # rename the columns for compatibility with the plotting function
            self.rename_columns("VT", {"Y_2w": "V2w", "X_1w": "V1w"})
            self.set_unit({"I": "uA", "V": "uV"})
            #self.df_plot_nonlinear(handlers=(ax[1],phi[1],ax[0],phi[0]))
            if out_range:
                print("out-range happened, rerun")
        except KeyboardInterrupt:
            print("Measurement interrupted")
        finally:
            tmp_df.to_csv(file_path, sep="\t", index=False, float_format="%.12f")
            if source == "sr830":
                meter_2w.sine_voltage = 0
            if source == "6221":
                source_6221.shutdown()

    #@print_help_if_needed
    #def measure_RT_SR830_ITC503(self, measurename_all, *var_tuple, resist: float) -> None:
    #    """
    #    Measure the Resist-Temperature relation using SR830 as both meter and source and store the data in the corresponding file(meters need to be loaded before calling this function, and the first is the source)

    #    Args:
    #        measurename_all (str): the full name of the measurement
    #        var_tuple (Tuple): the variables of the measurement, use "-h" to see the available options
    #        resist (float): the resistance of the resistor, used only to calculate corresponding voltage
    #    """
    #    file_path = self.get_filepath(measurename_all, *var_tuple)
    #    self.add_measurement(measurename_all)
    #    curr = MeasureManager.split_no_str(var_tuple[0])
    #    curr = factor(curr[1], "to_SI") * curr[0]  # [A]
    #    print(f"Filename is: {file_path.name}")
    #    print(f"Curr: {curr} A")
    #    print(f"estimated T range: {var_tuple[7]}-{var_tuple[8]} K")
    #    measure_delay = 0.5  # [s]
    #    frequency = 51.637  # [Hz]
    #    volt = curr * resist  # [V]

    #    self.setup_SR830()
    #    itc = self.instrs["itc"]
    #    meter1 = self.instrs["sr830"][0]
    #    meter2 = self.instrs["sr830"][1]
    #    print("====================")
    #    print(f"The first meter is {meter1.adapter}")
    #    print(f"Measuring {meter1.harmonic}-order signal")
    #    print("====================")
    #    print(f"The second meter is {meter2.adapter}")
    #    print(f"Measuring {meter2.harmonic}-order signal")
    #    print("====================")

    #    # increase voltage 0.02V/s to the needed value
    #    print(f"increasing voltage to targeted value {volt} V")
    #    amp = np.arange(0, volt, 0.01)
    #    for v in amp:
    #        meter1.sine_voltage = v
    #        time.sleep(0.5)
    #    print("voltage reached, start measurement")

    #    self.live_plot_init(2, 2, 2, 1000, 600, titles=[["XY-T", "phi"], ["XY-T", "phi"]],
    #                        axes_labels=[[[r"T (K)", r"V (V)"], ["T (K)", "phi"]],
    #                                     [[r"T (K)", r"V (V)"], ["T (K)", "phi"]]],
    #                        line_labels=[[["X", "Y"], ["", ""]], [["X", "Y"], ["", ""]]])
    #    meter1.reference_source = "Internal"
    #    meter1.frequency = frequency
    #    tmp_df = pd.DataFrame(columns=["T", "X1", "Y1", "R1", "phi1", "X2", "Y2", "R2", "phi2"])
    #    try:
    #        count = 0
    #        while True:
    #            count += 1
    #            time.sleep(measure_delay)
    #            list1 = meter1.snap("X", "Y", "R", "THETA")
    #            list2 = meter2.snap("X", "Y", "R", "THETA")
    #            temp = [itc.temperature]
    #            list_tot = temp + list1 + list2
    #            tmp_df.loc[len(tmp_df)] = list_tot

    #            self.live_plot_update(0, 0, 0, tmp_df["T"], tmp_df["X1"])
    #            self.live_plot_update(0, 0, 1, tmp_df["T"], tmp_df["Y1"])
    #            self.live_plot_update(0, 1, 0, tmp_df["T"], tmp_df["phi1"])
    #            self.live_plot_update(1, 0, 0, tmp_df["T"], tmp_df["X2"])
    #            self.live_plot_update(1, 0, 1, tmp_df["T"], tmp_df["Y2"])
    #            self.live_plot_update(1, 1, 0, tmp_df["T"], tmp_df["phi2"])
    #            if count % 10 == 0:
    #                tmp_df.to_csv(file_path, sep="\t", index=False)
    #    except KeyboardInterrupt:
    #        print("Measurement interrupted")
    #    finally:
    #        tmp_df.to_csv(file_path, sep="\t", index=False)
    #        meter1.sine_voltage = 0

    @print_help_if_needed
    def measure_VI_2182(self, measurename_all, *var_tuple, tmpfolder: str = None, delay: int = 5,
                        mode: Literal["0-max-0", "0--max-max-0"] = "0-max-0") -> None:
        """
        Measure the IV curve using Keithley 2182 to test the contacts. No file will be saved if test is True

        Args:
            measurename_all (str): the full name of the measurement
            var_tuple (Tuple): the variables of the measurement, use "-h" to see the available options
            tmpfolder (str): the temporary folder to store the data
            source (Literal["2182","6221"]): the source of the measurement
        """
        file_path = self.get_filepath(measurename_all, *var_tuple, tmpfolder=tmpfolder)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.add_measurement(measurename_all)
        curr = MeasureManager.split_no_str(var_tuple[0])
        curr = factor(curr[1], "to_SI") * curr[0]  # [A]
        print(f"Filename is: {file_path.name}")
        print(f"Max Curr: {curr} A")
        print(f"steps: {var_tuple[1] - 1}")
        curr_arr = np.linspace(0, curr, var_tuple[1])
        if mode == "0-max-0":
            curr_arr = np.concatenate((curr_arr, curr_arr[::-1]))
        elif mode == "0--max-max-0":
            curr_arr = np.concatenate((-curr_arr, -curr_arr[::-1], curr_arr, curr_arr[::-1]))
        measure_delay = delay  # [s]
        tmp_df = pd.DataFrame(columns=["I", "V", "T"])
        out_range = False

        instr_2182 = self.instrs["2182"]
        self.setup_2182()
        self.live_plot_init(1, 2, 1, 600, 1400,
                            titles=[["IV", "T"]],
                            axes_labels=[[["I (A)", "V (V)"], ["t", "T"]]],
                            line_labels=[[["", ""], ["", ""]]])

        source_6221 = self.instrs["6221"]
        self.setup_6221("dc")
        #source_6221.source_compliance = curr * 1E4  # compliance voltage
        source_6221.source_compliance = 5  # compliance voltage
        source_6221.source_range = curr / 0.6
        source_6221.source_current = 0
        source_6221.enable_source()
        try:
            for i, c in enumerate(curr_arr):
                source_6221.source_current = c
                time.sleep(measure_delay)
                tmp_df.loc[len(tmp_df)] = [c, instr_2182.voltage, self.instrs["itc"].temperature]
                self.live_plot_update(0, 0, 0, [tmp_df["I"]], [tmp_df["V"]])
                self.live_plot_update(0, 1, 0, [measure_delay * np.arange(len(tmp_df["T"]))], [tmp_df["T"]])
                if i % 10 == 0:
                    tmp_df.to_csv(file_path, sep="\t", index=False, float_format="%.12f")
            self.dfs["IV"] = tmp_df.copy()
            self.set_unit({"I": "uA", "V": "uV"})
        except KeyboardInterrupt:
            print("Measurement interrupted")
        finally:
            tmp_df.to_csv(file_path, sep="\t", index=False, float_format="%.12f")
            source_6221.disable_source()
            source_6221.shutdown()
            instr_2182.shutdown()

    @print_help_if_needed
    def measure_VI_SR830(self, measurename_all, *var_tuple, tmpfolder: str = None,
                         source: Literal["sr830", "6221"], delay: int = 15, offset_6221=0,
                         order: Tuple[int, int] = (1, 2)) -> None:
        """
        conduct the 1-pair nonlinear measurement using 2 SR830 meters and store the data in the corresponding file. Using first meter to measure 2w signal and also as the source if appoint SR830 as source. (meters need to be loaded before calling this function). When using Keithley 6221 current source, the max voltage is the compliance voltage and the resistance does not have specific meaning, just used for calculating the current.
        appoint the resistor to 1000 when using 6221(just for convenience, no special reason)

        Args:
            measurename_all (str): the full name of the measurement
            var_tuple (Tuple): the variables of the measurement, use "-h" to see the available options
            tmpfolder (str): the temporary folder to store the data
            source (Literal["sr830", "6221"]): the source of the measurement
            delay (int): the delay time between each measurement
            offset_6221 (int, [A]): the offset of the 6221 current source
        """
        file_path = self.get_filepath(measurename_all, *var_tuple, tmpfolder=tmpfolder)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if offset_6221 != 0:
            file_path = file_path.with_name(file_path.name + f"offset{offset_6221}")
        self.add_measurement(measurename_all)
        print(f"Filename is: {file_path.name}")
        print(f"Max Curr: {var_tuple[0] / var_tuple[1]} A")
        print(f"steps: {var_tuple[2] - 1}")
        print(f"2w meter: {self.instrs['sr830'][0].adapter}")
        print(f"1w meter: {self.instrs['sr830'][1].adapter}")
        amp = np.linspace(0, var_tuple[0], var_tuple[2])
        freq = var_tuple[3]
        resist = var_tuple[1]
        measure_delay = delay  # [s]
        tmp_df = pd.DataFrame(columns=["curr", "X_2w", "Y_2w", "R_2w", "phi_2w", "X_1w", "Y_1w", "R_1w", "phi_1w", "T"])
        out_range = False

        self.setup_SR830()
        meter_2w = self.instrs['sr830'][0]
        meter_1w = self.instrs['sr830'][1]
        meter_2w.harmonic = order[1]
        meter_1w.harmonic = order[0]

        self.live_plot_init(2, 2, 2, 600, 1400,
                            titles=[["2w", "phi"], ["1w", "phi"]],
                            axes_labels=[[["I (A)", "V2w (V)"], ["I (A)", "phi"]],
                                         [["I (A)", "V1w (V)"], ["I (A)", "phi"]]],
                            line_labels=[[["X", "Y"], ["", ""]], [["X", "Y"], ["", ""]]])
        if source == "sr830":
            meter_1w.reference_source_trigger = "POS EDGE"
            meter_2w.reference_source_trigger = "SINE"
            meter_2w.reference_source = "Internal"
            meter_2w.frequency = freq
        elif source == "6221":
            # 6221 use half peak-to-peak voltage as amplitude
            amp *= np.sqrt(2)
            source_6221 = self.instrs["6221"]
            self.setup_6221(offset=offset_6221)
            source_6221.source_compliance = amp[-1] + 0.1
            source_6221.source_range = amp[-1] / resist / 0.7
            print(f"Keithley 6221 source range is set to {amp[-1] / resist / 0.7} A")
            source_6221.waveform_frequency = freq
            meter_1w.reference_source_trigger = "POS EDGE"
            meter_2w.reference_source_trigger = "POS EDGE"
        try:
            for i, v in enumerate(amp):
                if source == "sr830":
                    meter_2w.sine_voltage = v
                elif source == "6221":
                    source_6221.waveform_abort()
                    source_6221.waveform_amplitude = v / resist
                    source_6221.waveform_arm()
                    source_6221.waveform_start()
                time.sleep(measure_delay)
                if meter_1w.is_out_of_range():
                    out_range = True
                if meter_2w.is_out_of_range():
                    out_range = True
                list_2w = meter_2w.snap("X", "Y", "R", "THETA")
                list_1w = meter_1w.snap("X", "Y", "R", "THETA")
                temp = self.instrs["itc"].temperature
                if source == "sr830":
                    list_tot = [v / resist] + list_2w + list_1w + [temp]
                if source == "6221":
                    list_tot = [v / resist / np.sqrt(2)] + list_2w + list_1w + [temp]
                print(
                    f"curr: {list_tot[0] * 1E6:.8f} uA\t 2w: {list_tot[1:5]}\t 1w: {list_tot[5:9]}\t T: {list_tot[-1]}")
                tmp_df.loc[len(tmp_df)] = list_tot
                self.live_plot_update([0, 0, 0, 1, 1, 1],
                                      [0, 0, 1, 0, 0, 1],
                                      [0, 1, 0, 0, 1, 0],
                                      [tmp_df["curr"]] * 6,
                                      np.array(tmp_df[["X_2w", "Y_2w", "phi_2w", "X_1w", "Y_1w", "phi_1w"]]).T)
                if i % 10 == 0:
                    tmp_df.to_csv(file_path, sep="\t", index=False, float_format="%.12f")
            self.dfs["nonlinear"] = tmp_df.copy()
            # rename the columns for compatibility with the plotting function
            self.rename_columns("nonlinear", {"Y_2w": "V2w", "X_1w": "V1w"})
            self.set_unit({"I": "uA", "V": "uV"})
            if out_range:
                print("out-range happened, rerun")
        except KeyboardInterrupt:
            print("Measurement interrupted")
        finally:
            tmp_df.to_csv(file_path, sep="\t", index=False, float_format="%.12f")
            if source == "sr830":
                meter_2w.sine_voltage = 0
            if source == "6221":
                source_6221.shutdown()

    @staticmethod
    def get_visa_resources() -> Tuple[str, ...]:
        """
        return a list of visa resources
        """
        return pyvisa.ResourceManager().list_resources()

    @staticmethod
    def write_header(file_path: Path, header: str) -> None:
        """
        write the header to the file

        Args:
            file_path (str): the file path
            header (str): the header to write
        """
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(header)

    @staticmethod
    def split_no_str(s: str) -> Tuple[float | None, str | None]:
        """
        split the string into the string part and the float part.

        Args:
            s (str): the string to split

        Returns:
            Tuple[float,str]: the string part and the integer part
        """
        match = re.match(r"([0-9.]+)([a-zA-Z]+)", s, re.I)

        if match:
            items = match.groups()
            return float(items[0]), items[1]
        else:
            return None, None

    def print_pairs(self, sub_type, v1_2w_meter: Literal[0, 1] = 0, v2_1w_meter: Literal[0, 1] = 1):
        if "1pair" in sub_type.split("-"):
            print("===========================================")
            print(f"2w meter: {self.instrs['sr830'][v1_2w_meter].adapter}")
            print(f"1w meter: {self.instrs['sr830'][v2_1w_meter].adapter}")
            print("===========================================")
        elif "2pair" in sub_type.split("-"):
            print("===========================================")
            print(f"V1 meter: {self.instrs['sr830'][v1_2w_meter].adapter}\t ORDER: {self.instrs['sr830'][0].harmonic}")
            print(f"V2 meter: {self.instrs['sr830'][v2_1w_meter].adapter}\t ORDER: {self.instrs['sr830'][1].harmonic}")
            print("===========================================")

    @staticmethod
    def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='#',
                           print_end="\r") -> None:
        """
        Call in a loop to create terminal progress bar

        Args:
            iteration (int): current iteration
            total (int): total iterations
            prefix (str): prefix string
            suffix (str): suffix string
            decimals (int): positive number of decimals in percent complete
            length (int): character length of bar
            fill (str): bar fill character
            print_end (str): end character (e.g. "\r", "\r\n")
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        barr = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} [{barr}] {percent}% {suffix}', end=print_end, flush=True)
        # Print New Line on Complete
        if iteration == total:
            print()


class ITC(ABC, DataPlot):
    # parent class to incorporate both two ITCs
    @property
    @abstractmethod
    def temperature(self):
        """return the precise temperature of the sample"""
        pass

    @abstractmethod
    def set_temperature(self, temp):
        """
        set the target temperature for sample, as for other parts' temperature, use the methods for each ITC
        """
        pass

    @property
    @abstractmethod
    def pid(self):
        """
        return the PID parameters
        """
        pass

    @abstractmethod
    def set_pid(self, pid_dict):
        """
        set the PID parameters
        
        Args:
            pid_dict (Dict): a dictionary as {"P": float, "I": float, "D": float}
        """
        pass

    @abstractmethod
    def correction_ramping(self, temp: float, trend: Literal["up", "down", "up-huge", "down-huge"]):
        """
        Correct the sensor choosing or pressure when ramping through the temperature threshold

        Args:
            temp (float): the current temperature
            trend (Literal["up","down"]): the trend of the temperature
        """
        pass

    def wait_for_temperature(self, temp, *, if_plot=False, delta=0.01, check_interval=1, stability_counter=120,
                             thermalize_counter=120):
        """
        wait for the temperature to stablize for a certain time length

        Args: temp (float): the target temperature delta (float): the temperature difference to consider the
        temperature stablized check_interval (int,[s]): the interval to check the temperature stability_counter (
        int): the number of times the temperature is within the delta range to consider the temperature stablized
        thermalize_counter (int): the number of times to thermalize the sample if_plot (bool): whether to plot the
        temperature change
        """
        if self.temperature < temp - 100:
            trend = "up-huge"
        elif self.temperature > temp + 100:
            trend = "down-huge"
        elif self.temperature < temp:
            trend = "up"
        else:
            trend = "down"

        if if_plot:
            self.live_plot_init(1, 1, 1, 600, 1400, titles=[["T ramping"]],
                                axes_labels=[[[r"Time (s)", r"T (K)"]]])
            t_arr = [0]
            T_arr = [self.temperature]
        i = 0
        while i < stability_counter:
            self.correction_ramping(self.temperature, trend)
            if abs(self.temperature - temp) < ITC.dynamic_delta(temp, delta):
                i += 1
            elif i >= 5:
                i -= 5
            if if_plot:
                t_arr.append(t_arr[-1] + check_interval)
                T_arr.append(self.temperature)
                self.live_plot_update(0, 0, 0, t_arr, T_arr)
            MeasureManager.print_progress_bar(i, stability_counter, prefix="Stablizing",
                                              suffix=f"Temperature: {self.temperature:.2f} K")
            time.sleep(check_interval)
        print("Temperature stablized")
        for i in range(thermalize_counter):
            MeasureManager.print_progress_bar(i+1, thermalize_counter, prefix="Thermalizing",
                                              suffix=f"Temperature: {self.temperature:.2f} K")
            if if_plot:
                t_arr.append(t_arr[-1] + check_interval)
                T_arr.append(self.temperature)
                self.live_plot_update(0, 0, 0, t_arr, T_arr)
            time.sleep(check_interval)
        print("Thermalizing finished")

    def ramp_to_temperature(self, temp, *, delta=0.01, check_interval=1, stability_counter=120, thermalize_counter=120,
                            pid=None, ramp_rate=None, wait=True, if_plot=False):
        """ramp temperature to the target value (not necessary sample temperature)"""
        self.set_temperature(temp)
        if pid is not None:
            self.set_pid(pid)
        if wait:
            self.wait_for_temperature(temp, delta=delta, check_interval=check_interval,
                                      stability_counter=stability_counter,
                                      thermalize_counter=thermalize_counter, if_plot=if_plot)

    @staticmethod
    def dynamic_delta(temp, delta_lowt) -> float:
        """
        calculate a dynamic delta to help high temperature to stabilize (reach 0.1K tolerance when 300K and {delta_lowt} when 10K)
        """
        # let the delta be delta_lowt at 1.5K and 0.2K at 300K
        t_low = 1.5
        delta_hight = 0.2
        t_high = 300
        return (delta_hight - delta_lowt) * (temp - t_low) / (t_high - t_low) + delta_lowt


class ITCMercury(ITC):
    def __init__(self, proj_name: str, address="TCPIP0::10.97.27.13::7020::SOCKET"):
        self.mercury = MercuryITC("mercury_itc", address)

    @property
    def pres(self):
        return self.mercury.pressure()

    def set_pres(self, pres: float):
        self.mercury.pressure_setpoint(pres)

    @property
    def flow(self):
        return self.mercury.gas_flow()

    def set_flow(self, flow: float):
        """
        set the gas flow, note the input value is percentage, from 0 to 99.9 (%)
        """
        if not 0.0 < flow < 100.0:
            raise ValueError("Flow must be between 0.0 and 100.0 (%)")
        self.mercury.gas_flow(flow)

    @property
    def pid(self):
        return {"P": self.mercury.temp_loop_P(), "I": self.mercury.temp_loop_I(),
                "D": self.mercury.temp_loop_D()}

    def set_pid(self, pid: dict):
        """
        set the pid of probe temp loop
        """
        self.mercury.temp_PID_auto("OFF")
        self.mercury.temp_PID = (pid["P"], pid["I"], pid["D"])

    @property
    def temperature(self):
        return self.mercury.probe_temp()

    def set_temperature(self, temp, vti_temp=None):
        """set the target temperature for sample"""
        self.mercury.temp_setpoint(temp)
        if vti_temp is not None:
            self.mercury.vti_temp_setpoint(vti_temp)
        else:
            self.mercury.vti_temp_setpoint(self.mercury.calculate_vti_temp(temp))

    @property
    def vti_temperature(self):
        return self.mercury.vti_temp()

    def set_vti_temperature(self, temp):
        self.mercury.vti_temp_setpoint(temp)

    def ramp_to_temperature(self, temp, *, delta=0.01, check_interval=1, stability_counter=120, thermalize_counter=120,
                            pid=None, ramp_rate=None, wait=True, if_plot=False):
        """ramp temperature to the target value (not necessary sample temperature) Args: temp (float): the target
        temperature delta (float): the temperature difference to consider the temperature stablized check_interval (
        int,[s]): the interval to check the temperature stability_counter (int): the number of times the temperature
        is within the delta range to consider the temperature stablized thermalize_counter (int): the number of times
        to thermalize the sample pid (Dict): a dictionary as {"P": float, "I": float, "D": float} ramp_rate (float,
        [K/min]): the rate to ramp the temperature
        """
        self.set_temperature(temp)
        if pid is not None:
            self.set_pid(pid)

        if ramp_rate is not None:
            self.mercury.probe_ramp_rate(ramp_rate)
            # self.mercury.vti_heater_rate(ramp_rate)
            self.mercury.probe_temp_ramp_mode("ON")
        else:
            self.mercury.probe_temp_ramp_mode("OFF")
        if wait:
            self.wait_for_temperature(temp, delta=delta, check_interval=check_interval,
                                      stability_counter=stability_counter,
                                      thermalize_counter=thermalize_counter, if_plot=if_plot)

    def correction_ramping(self, temp: float, trend: Literal["up", "down", "up-huge", "down-huge"]):
        """
        Correct the sensor choosing or pressure when ramping through the temperature threshold

        Args:
            temp (float): the current temperature
            trend (Literal["up","down","up-huge","down-huge"]): the trend of the temperature
        """
        if trend == "up-huge":
            self.set_pres(5)
        elif trend == "down-huge":
            if temp >= 5:
                self.set_pres(15)
            else:
                self.set_pres(3)
        else:
            self.set_pres(3)


class ITCs(ITC):
    """ Represents the ITC503 Temperature Controllers and provides a high-level interface for interacting with the instruments. 
    
    There are two ITC503 incorporated in the setup, named up and down. The up one measures the temperature of the heat switch(up R1), PT2(up R2), leaving R3 no specific meaning. The down one measures the temperature of the sorb(down R1), POT LOW(down R2), POT HIGH(down R3).
    """

    def __init__(self, address_up="GPIB0::23::INSTR", address_down="GPIB0::24::INSTR", clear_buffer=True):
        self.itc_up = ITC503(address_up, clear_buffer=clear_buffer)
        self.itc_down = ITC503(address_down, clear_buffer=clear_buffer)

    def chg_display(self, itc_name, target):
        """
        This function is used to change the front display of the ITC503

        Parameters: itc_name (str): The name of the ITC503, "up" or "down" or "all" target (str):  'temperature
        setpoint', 'temperature 1', 'temperature 2', 'temperature 3', 'temperature error', 'heater',
        'heater voltage', 'gasflow', 'proportional band', 'integral action time', 'derivative action time',
        'channel 1 freq/4', 'channel 2 freq/4', 'channel 3 freq/4'.

        Returns:
        None
        """
        if itc_name == "all":
            self.itc_up.front_panel_display = target
            self.itc_down.front_panel_display = target
        elif itc_name == "up":
            self.itc_up.front_panel_display = target
        elif itc_name == "down":
            self.itc_down.front_panel_display = target

    def chg_pointer(self, itc_name, target: tuple):
        """
        used to change the pointer of the ITCs

        Parameters: itc_name (str): The name of the ITC503, "up" or "down" or "all" target (tuple): A tuple property
        to set pointers into tables for loading and examining values in the table, of format (x, y). The significance
        and valid values for the pointer depends on what property is to be read or set. The value for x and y can be
        in the range 0 to 128.

        Returns:
        None
        """
        if itc_name == "all":
            self.itc_up.pointer = target
            self.itc_down.pointer = target
        elif itc_name == "up":
            self.itc_up.pointer = target
        elif itc_name == "down":
            self.itc_down.pointer = target

    def set_temperature(self, temp):
        """
        set the target temperature for sample, as for other parts' temperature, use the methods for each ITC

        Args:
            temp (float): the target temperature
            itc_name (Literal["up","down","all"]): the ITC503 to set the temperature
        """
        self.itc_down.temperature_setpoint = temp

    def ramp_to_temperature_selective(self, temp, itc_name: Literal["up", "down"], P=None, I=None, D=None):
        """
        used to ramp the temperature of the ITCs, this method will wait for the temperature to stablize and thermalize for a certain time length
        """
        self.control_mode = ("RU", itc_name)
        if itc_name == "up":
            itc_here = self.itc_up
        if itc_name == "down":
            itc_here = self.itc_down
        itc_here.temperature_setpoint = temp
        if P is not None and I is not None and D is not None:
            itc_here.auto_pid = False
            itc_here.proportional_band = P
            itc_here.integral_action_time = I
            itc_here.derivative_action_time = D
        else:
            itc_here.auto_pid = True
        itc_here.heater_gas_mode = "AM"
        print(f"temperature setted to {temp}")

    @property
    def version(self):
        """ Returns the version of the ITC503. """
        return [self.itc_up.version, self.itc_down.version]

    @property
    def control_mode(self):
        """ Returns the control mode of the ITC503. """
        return [self.itc_up.control_mode, self.itc_down.control_mode]

    @control_mode.setter
    def control_mode(self, mode: Tuple[Literal["LU", "RU", "LL", "RL"], Literal["all", "up", "down"]]):
        """ Sets the control mode of the ITC503. A two-element list is required. The second elecment is "all" or "up"
        or "down" to specify which ITC503 to set."""
        if mode[1] == "all":
            self.itc_up.control_mode = mode[0]
            self.itc_down.control_mode = mode[0]
        elif mode[1] == "up":
            self.itc_up.control_mode = mode[0]
        elif mode[1] == "down":
            self.itc_down.control_mode = mode[0]

    @property
    def heater_gas_mode(self):
        """ Returns the heater gas mode of the ITC503. """
        return [self.itc_up.heater_gas_mode, self.itc_down.heater_gas_mode]

    @heater_gas_mode.setter
    def heater_gas_mode(self, mode: Tuple[Literal["MANUAL", "AM", "MA", "AUTO"], Literal["all", "up", "down"]]):
        """ Sets the heater gas mode of the ITC503. A two-element list is required. The second elecment is "all" or
        "up" or "down" to specify which ITC503 to set."""
        if mode[1] == "all":
            self.itc_up.heater_gas_mode = mode[0]
            self.itc_down.heater_gas_mode = mode[0]
        elif mode[1] == "up":
            self.itc_up.heater_gas_mode = mode[0]
        elif mode[1] == "down":
            self.itc_down.heater_gas_mode = mode[0]

    @property
    def heater_power(self):
        """ Returns the heater power of the ITC503. """
        return [self.itc_up.heater, self.itc_down.heater]

    @property
    def heater_voltage(self):
        """ Returns the heater voltage of the ITC503. """
        return [self.itc_up.heater_voltage, self.itc_down.heater_voltage]

    @property
    def gas_flow(self):
        """ Returns the gasflow of the ITC503. """
        return [self.itc_up.gasflow, self.itc_down.gasflow]

    @property
    def proportional_band(self):
        """ Returns the proportional band of the ITC503. """
        return [self.itc_up.proportional_band, self.itc_down.proportional_band]

    @property
    def integral_action_time(self):
        """ Returns the integral action time of the ITC503. """
        return [self.itc_up.integral_action_time, self.itc_down.integral_action_time]

    @property
    def derivative_action_time(self):
        """ Returns the derivative action time of the ITC503. """
        return [self.itc_up.derivative_action_time, self.itc_down.derivative_action_time]

    def set_pid(self, pid: dict, mode: Literal["all", "up", "down"] = "down"):
        """ Sets the PID of the ITC503. A three-element list is required. The second elecment is "all" or "up" or "down" to specify which ITC503 to set. 
        The P,I,D here are the proportional band (K), integral action time (min), and derivative action time(min), respectively.
        """
        self.control_mode = ("RU", mode)
        if mode == "all":
            self.itc_up.proportional_band = pid["P"]
            self.itc_down.proportional_band = pid["P"]
            self.itc_up.integral_action_time = pid["I"]
            self.itc_down.integral_action_time = pid["I"]
            self.itc_up.derivative_action_time = pid["D"]
            self.itc_down.derivative_action_time = pid["D"]
        if mode == "up":
            self.itc_up.proportional_band = pid["P"]
            self.itc_up.integral_action_time = pid["I"]
            self.itc_up.derivative_action_time = pid["D"]
        if mode == "down":
            self.itc_down.proportional_band = pid["P"]
            self.itc_down.integral_action_time = pid["I"]
            self.itc_down.derivative_action_time = pid["D"]

        if self.itc_up.proportional_band == 0:
            return ""
        return f"{mode} PID(power percentage): 100*(E/{pid['P']}+E/{pid['P']}*t/60{pid['I']}-dE*60{pid['D']}/{pid['P']}), [K,min,min]"

    @property
    def auto_pid(self):
        """ Returns the auto pid of the ITC503. """
        return [self.itc_up.auto_pid, self.itc_down.auto_pid]

    @auto_pid.setter
    def auto_pid(self, mode):
        """ Sets the auto pid of the ITC503. A two-element list is required. The second elecment is "all" or "up" or
        "down" to specify which ITC503 to set."""
        if mode[1] == "all":
            self.itc_up.auto_pid = mode[0]
            self.itc_down.auto_pid = mode[0]
        elif mode[1] == "up":
            self.itc_up.auto_pid = mode[0]
        elif mode[1] == "down":
            self.itc_down.auto_pid = mode[0]

    @property
    def sweep_status(self):
        """ Returns the sweep status of the ITC503. """
        return [self.itc_up.sweep_status, self.itc_down.sweep_status]

    @property
    def temperature_setpoint(self):
        """ Returns the temperature setpoint of the ITC503. """
        return [self.itc_up.temperature_setpoint, self.itc_down.temperature_setpoint]

    @temperature_setpoint.setter
    def temperature_setpoint(self, temperature):
        """ Sets the temperature setpoint of the ITC503. A two-element list is required. The second elecment is "all"
        or "up" or "down" to specify which ITC503 to set."""
        if temperature[1] == "all":
            self.itc_up.temperature_setpoint = temperature[0]
            self.itc_down.temperature_setpoint = temperature[0]
        elif temperature[1] == "up":
            self.itc_up.temperature_setpoint = temperature[0]
        elif temperature[1] == "down":
            self.itc_down.temperature_setpoint = temperature[0]

    @property
    def temperatures(self):
        """ Returns the temperatures of the whole device as a dict. """
        return {"sw": self.itc_up.temperature_1, "pt2": self.itc_up.temperature_2, "sorb": self.itc_down.temperature_1,
                "pot_low": self.itc_down.temperature_2, "pot_high": self.itc_down.temperature_3}

    @property
    def temperature(self):
        """ Returns the precise temperature of the sample """
        if self.temperatures["pot_high"] < 1.9:
            return self.temperatures["pot_low"]
        elif self.temperatures["pot_high"] >= 1.9:
            return self.temperatures["pot_high"]
