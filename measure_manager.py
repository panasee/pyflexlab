#!/usr/bin/env python

"""
This module is responsible for managing the measure-related folders and data
"""
from typing import List, Tuple, Literal
import time
import numpy as np
from pymeasure.instruments.srs import SR830
from pymeasure.instruments.oxfordinstruments import ITC503
import pyvisa
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re


from common.file_organizer import FileOrganizer
from common.data_plot import DataPlot
from common.constants import factor


def print_help_if_needed(func: callable) -> callable:
    """decorator used to print the help message if the first argument is '-h'"""
    def wrapper(self,measurename_all, *var_tuple, **kwargs):
        if var_tuple[0] == "-h":
            measure_name,_ = FileOrganizer.measurename_decom(measurename_all)
            print(FileOrganizer.query_namestr(measure_name))
            return None
        return func(self, measurename_all,*var_tuple, **kwargs)
    return wrapper

class MeasureManager(FileOrganizer):
    """This class is a subclass of FileOrganizer and is responsible for managing the measure-related folders and data"""

    def __init__(self, proj_name: str) -> None:
        """Note that the FileOrganizer.out_database_init method should be called to assign the correct path to the out_database attribute. This method should be called before the MeasureManager object is created."""
        super().__init__(proj_name) # Call the constructor of the parent class
        self.instrs = {}
        # load params for plotting in measurement
        DataPlot.load_settings(False, False)

    def load_SR830(self, *addresses: List[str]) -> None:
        """
        load SR830 instruments according the addresses, store them in self.sr830 in corresponding order

        Args:
            addresses (List[str]): the addresses of the SR830 instruments (take care of the order)
        """
        self.instrs["sr830"] = []
        for addr in addresses:
            self.instrs["sr830"].append(SR830(addr))

    def load_ITC503(self, gpib_up: str, gpib_down: str) -> None:
        """
        load ITC503 instruments according the addresses, store them in self.itc503 in corresponding order

        Args:
            addresses (List[str]): the addresses of the ITC503 instruments (take care of the order)
        """
        self.instrs["itc503"] = ITCs(gpib_up, gpib_down)
    
    def setup_SR830(self) -> None:
        """
        setup the SR830 instruments using pre-stored setups here, this function will not fully reset the instruments, only overwrite the specific settings here, other settings will all be reserved
        """
        for instr in self.instrs["sr830"]:
            instr.filter_slope = 24
            instr.time_constant = 0.3
            instr.input_config = "A-B"
            instr.input_coupling = "AC"
            instr.sine_voltage = 0

    @print_help_if_needed
    def measure_RT_SR830_ITC503(self, measurename_all, *var_tuple, resist: float) -> None:
        """
        Measure the Resist-Temperature relation using SR830 as both meter and source and store the data in the corresponding file(meters need to be loaded before calling this function, and the first is the source)

        Args:
            measurename_all (str): the full name of the measurement
            var_tuple (Tuple): the variables of the measurement, use "-h" to see the available options
            resist (float): the resistance for the source, used only to calculate corresponding voltage
        """
        file_path = self.get_filepath(measurename_all,*var_tuple)
        self.add_measurement(measurename_all)
        curr = MeasureManager.split_no_str(var_tuple[0])
        curr = factor(curr[1], "to_SI") * curr[0] # [A]
        print(f"Filename is: {file_path.name}")
        print(f"Curr: {curr} A")
        print(f"estimated T range: {var_tuple[7]}-{var_tuple[8]} K")
        measure_delay = 0.5 # [s]
        frequency = 51.637 # [Hz]
        volt = curr * resist # [V]
        conversion_T = 5 # [K], the temperature for changing between pot_low and pot_high

        self.setup_SR830()
        itc503 = self.instrs["itc503"]
        meter1 = self.instrs["sr830"][0]
        meter2 = self.instrs["sr830"][1]
        print("====================")
        print(f"The first meter is {meter1.adapter}")
        print(f"Measuring {meter1.harmonic}-order signal")
        print("====================")
        print(f"The second meter is {meter2.adapter}")
        print(f"Measuring {meter2.harmonic}-order signal")
        print("====================")

        # increase voltage 0.02V/s to the needed value
        print(f"increasing voltage to targeted value {volt} V")
        amp = np.arange(0, volt, 0.01)
        for v in amp:
            meter1.sine_voltage = v
            time.sleep(0.5)
        print("voltage reached, start measurement")

        fig, ax = DataPlot.init_canvas(1,2,10,6)
        phi = [i.twinx() for i in ax]
        meter1.reference_source = "Internal"
        meter1.frequency = frequency
        tmp_df = pd.DataFrame(columns=["pot_low","pot_high","X1","Y1","R1","phi1","X2","Y2","R2","phi2"])
        try:
            count = 0
            while True:
                count += 1
                time.sleep(measure_delay)
                list1 = meter1.snap("X","Y","R","THETA")
                list2 = meter2.snap("X","Y","R","THETA")
                temp = [itc503.temperatures["pot_low"], itc503.temperatures["pot_high"]]
                list_tot = temp + list1 + list2
                tmp_df.loc[len(tmp_df)] = list_tot

                if var_tuple[-3] > conversion_T or var_tuple[-2] > conversion_T:
                    temp_str = "pot_high"
                else:
                    temp_str = "pot_low"
                ax[0].plot(tmp_df[temp_str],tmp_df["X1"],label="X1")
                ax[0].plot(tmp_df[temp_str],tmp_df["Y1"],label="Y1")
                phi[0].plot(tmp_df[temp_str],tmp_df["phi1"],label="phi1")
                ax[0].legend()
                ax[1].plot(tmp_df[temp_str],tmp_df["X2"],label="X2")
                ax[1].plot(tmp_df[temp_str],tmp_df["Y2"],label="Y2")
                phi[1].plot(tmp_df[temp_str],tmp_df["phi2"],label="phi2")
                ax[1].legend()
                plt.draw()
                if count % 10 == 0:
                    tmp_df.to_csv(file_path, index=False)
        except KeyboardInterrupt:
            print("Measurement interrupted")
        finally:
            tmp_df.to_csv(file_path, index=False)
            meter1.sine_voltage = 0
                

    @print_help_if_needed
    def measure_nonlinear_SR830(self,measurename_all, *var_tuple, tmpfolder:str = None, source : Literal["sr830", "6221"]) -> None:
        """
        conduct the 1-pair nonlinear measurement using 2 SR830 meters and store the data in the corresponding file. Using first meter to measure 2w signal and also as the source if appoint SR830 as source. (meters need to be loaded before calling this function)

        Args:
            measurename_all (str): the full name of the measurement
            var_tuple (Tuple): the variables of the measurement, use "-h" to see the available options
            tmpfolder (str): the temporary folder to store the data
            source (Literal["sr830", "6221"]): the source of the measurement
        """
        file_path = self.get_filepath(measurename_all,*var_tuple, tmpfolder)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.add_measurement(measurename_all)
        print(f"Filename is: {file_path.name}")
        print(f"Max Curr: {var_tuple[0]/var_tuple[1]} A")
        print(f"steps: {var_tuple[2]-1}")
        print(f"2w meter: {self.instrs['sr830'][0].adapter}")
        print(f"1w meter: {self.instrs['sr830'][1].adapter}")
        amp = np.linspace(0, var_tuple[0], var_tuple[2])
        freq = var_tuple[3]
        resist = var_tuple[1]
        measure_delay = 3 # [s]
        tmp_df = pd.DataFrame(columns=["curr","X_2w","Y_2w","R_2w","phi_2w","X_1w","Y_1w","R_1w","phi_1w"])

        self.setup_SR830()
        meter_2w = self.instrs['sr830'][0]
        meter_1w = self.instrs['sr830'][1]
        meter_2w.harmonic = 2
        meter_1w.harmonic = 1

        fig, ax = DataPlot.init_canvas(1,2,10,6)
        phi = [i.twinx() for i in ax]
        if source == "sr830":
            meter_2w.reference_source = "Internal"
            meter_2w.frequency = freq
            try:
                for i, v in enumerate(amp):
                    meter_2w.sine_voltage = v
                    time.sleep(measure_delay)
                    list_2w = meter_2w.snap("X","Y","R","THETA")
                    list_1w = meter_1w.snap("X","Y","R","THETA")
                    list_tot = [v/resist] + list_2w + list_1w
                    tmp_df.loc[len(tmp_df)] = list_tot
                    ax[0].plot(tmp_df["curr"],tmp_df["Y_2w"],label="2w")
                    phi[0].plot(tmp_df["curr"],tmp_df["phi_2w"],label="2w")
                    ax[1].plot(tmp_df["curr"],tmp_df["R_1w"],label="1w")
                    phi[1].plot(tmp_df["curr"],tmp_df["phi_1w"],label="1w")
                    plt.draw()
                    if i % 10 == 0:
                        tmp_df.to_csv(file_path, index=False)
            except KeyboardInterrupt:
                print("Measurement interrupted")
            finally:
                tmp_df.to_csv(file_path, index=False)
                meter_2w.sine_voltage = 0

        elif source == "6221":
            ##TODO: add the 6221 source##
            pass

    @staticmethod
    def get_visa_resources() -> Tuple[str]:
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
        with open(file_path, "w") as f:
            f.write(header)
    
    @staticmethod
    def split_no_str(s: str) -> Tuple[float, str]:
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



class ITCs():
    """ Represents the ITC503 Temperature Controllers and provides a high-level interface for interacting with the instruments. 
    
    There are two ITC503 incorporated in the setup, named up and down. The up one measures the temperature of the heat switch(up R1), PT2(up R2), leaving R3 no specific meaning. The down one measures the temperature of the sorb(down R1), POT LOW(down R2), POT HIGH(down R3).
    """

    def __init__(self, address_up="GPIB0::23::INSTR", address_down="GPIB0::24::INSTR",clear_buffer=True):
        self.itc_up = ITC503(address_up,clear_buffer=clear_buffer)
        self.itc_down = ITC503(address_down,clear_buffer=clear_buffer)

    def chg_display(self,itc_name, target):
        """
        This function is used to change the front display of the ITC503

        Parameters:
        itc_name (str): The name of the ITC503, "up" or "down" or "all"
        target (str):  ‘temperature setpoint’, ‘temperature 1’, ‘temperature 2’, ‘temperature 3’, ‘temperature error’, ‘heater’, ‘heater voltage’, ‘gasflow’, ‘proportional band’, ‘integral action time’, ‘derivative action time’, ‘channel 1 freq/4’, ‘channel 2 freq/4’, ‘channel 3 freq/4’.

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

    def chg_pointer(self, itc_name, target:tuple):
        """
        used to change the pointer of the ITCs

        Parameters:
        itc_name (str): The name of the ITC503, "up" or "down" or "all"
        target (tuple): A tuple property to set pointers into tables for loading and examining values in the table, of format (x, y). The significance and valid values for the pointer depends on what property is to be read or set. The value for x and y can be in the range 0 to 128. 

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

    def wait_for_temperature(self, itc_name, tempe):
        if itc_name == "up":
            self.itc_up.temperature_setpoint = tempe
            self.itc_up.wait_for_temperature(tempe)
        if itc_name == "down":
            self.itc_down.temperature_setpoint = tempe
            self.itc_down.wait_for_temperature(tempe)

    def chg_sensor(self, target:str):
        """
        used to change the sensor of the ITCs for heater control

        Parameters:
        target (str): "sw", "pt2", "sorb", "pot_low", "pot_high"

        Returns:
        None
        """
        #if target == "sw":
        #    self.itc_up.write("H1")
        #if target == "pt2":
        #    self.itc_up.write("H2")
        #if target == "sorb":
        #    self.itc_down.write("H1")
        #if target == "pot_low":
        #    self.itc_down.write("H2")
        #if target == "pot_high":
        #    self.itc_down.write("H3")
        print("this function is not implemented yet.")

    @property
    def version(self):
        """ Returns the version of the ITC503. """
        return [self.itc_up.version,self.itc_down.version]

    @property
    def control_mode(self):
        """ Returns the control mode of the ITC503. """
        return [self.itc_up.control_mode,self.itc_down.control_mode]
    
    @control_mode.setter
    def control_mode(self,mode):
        """ Sets the control mode of the ITC503. A two-element list is required. The second elecment is "all" or "up" or "down" to specify which ITC503 to set. """
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
        return [self.itc_up.heater_gas_mode,self.itc_down.heater_gas_mode]

    @heater_gas_mode.setter
    def heater_gas_mode(self,mode):
        """ Sets the heater gas mode of the ITC503. A two-element list is required. The second elecment is "all" or "up" or "down" to specify which ITC503 to set. """
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
        return [self.itc_up.heater,self.itc_down.heater]

    @property
    def heater_voltage(self):
        """ Returns the heater voltage of the ITC503. """
        return [self.itc_up.heater_voltage,self.itc_down.heater_voltage]

    @property
    def gas_flow(self):
        """ Returns the gasflow of the ITC503. """
        return [self.itc_up.gasflow,self.itc_down.gasflow]

    @property
    def proportional_band(self):
        """ Returns the proportional band of the ITC503. """
        return [self.itc_up.proportional_band,self.itc_down.proportional_band]

    @property
    def integral_action_time(self):
        """ Returns the integral action time of the ITC503. """
        return [self.itc_up.integral_action_time,self.itc_down.integral_action_time]    

    @property
    def derivative_action_time(self):
        """ Returns the derivative action time of the ITC503. """
        return [self.itc_up.derivative_action_time,self.itc_down.derivative_action_time]

    def set_pid(self, P, I, D, mode="all"):
        """ Sets the PID of the ITC503. A three-element list is required. The second elecment is "all" or "up" or "down" to specify which ITC503 to set. 
        The P,I,D here are the proportional band (K), integral action time (min), and derivative action time(min), respectively.
        """
        if mode == "all":
            self.itc_up.proportional_band = P
            self.itc_down.proportional_band = P
            self.itc_up.integral_action_time = I
            self.itc_down.integral_action_time = I
            self.itc_up.derivative_action_time = D
            self.itc_down.derivative_action_time = D
        if mode == "up":
            self.itc_up.proportional_band = P
            self.itc_up.integral_action_time = I
            self.itc_up.derivative_action_time = D
        if mode == "down":
            self.itc_down.proportional_band = P
            self.itc_down.integral_action_time = I
            self.itc_down.derivative_action_time = D
        
        if self.itc_up.proportional_band == 0:
            return ""
        return f"{mode} PID(power percentage): 100*(E/{P}+E/{P}*t/60{I}-dE*60{D}/{P}), [K,min,min]"

    
    @property
    def auto_pid(self):
        """ Returns the auto pid of the ITC503. """
        return [self.itc_up.auto_pid,self.itc_down.auto_pid]
    
    @auto_pid.setter
    def auto_pid(self,mode):
        """ Sets the auto pid of the ITC503. A two-element list is required. The second elecment is "all" or "up" or "down" to specify which ITC503 to set. """
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
        return [self.itc_up.sweep_status,self.itc_down.sweep_status]

    @property
    def temperature_setpoint(self):
        """ Returns the temperature setpoint of the ITC503. """
        return [self.itc_up.temperature_setpoint,self.itc_down.temperature_setpoint]
    
    @temperature_setpoint.setter
    def temperature_setpoint(self,temperature):
        """ Sets the temperature setpoint of the ITC503. A two-element list is required. The second elecment is "all" or "up" or "down" to specify which ITC503 to set. """
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
        return {"sw":self.itc_up.temperature_1,"pt2":self.itc_up.temperature_2,"sorb":self.itc_down.temperature_1,"pot_low":self.itc_down.temperature_2,"pot_high":self.itc_down.temperature_3}
