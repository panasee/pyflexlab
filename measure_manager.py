#!/usr/bin/env python

"""
This module is responsible for managing the measure-related folders and data
Note each instrument better be initialzed right before the measurement, as there may be a long time between loading and measuremnt, leading to possibilities of parameter changing
"""
from typing import List, Tuple, Literal
import time
import numpy as np
from pymeasure.instruments.srs import SR830
from pymeasure.instruments.oxfordinstruments import ITC503
from pymeasure.instruments.keithley import Keithley2400
from pymeasure.instruments.keithley import Keithley6221
#from pymeasure.instruments.keithley import Keithley2182
import pyvisa
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
#import qcodes as qc
from qcodes.instrument_drivers.oxford import OxfordMercuryiPS


from common.file_organizer import print_help_if_needed
from common.data_plot import DataPlot
from common.constants import factor
from common.mercury_itc_driver import MercuryITC

class MeasureManager(DataPlot):
    """This class is a subclass of FileOrganizer and is responsible for managing the measure-related folders and data"""

    def __init__(self, proj_name: str) -> None:
        """Note that the FileOrganizer.out_database_init method should be called to assign the correct path to the out_database attribute. This method should be called before the MeasureManager object is created."""
        super().__init__(proj_name) # Call the constructor of the parent class
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

#    def load_2182(self, address: str) -> None:
#        """
#        load Keithley 2182 instrument according to the address, store it in self.instrs["2182"]
#        """
#        self.instrs["2182"] = Keithley2182(address)

    def load_2400(self, address: str) -> None:
        """
        load Keithley 2400 instrument according to the address, store it in self.instrs["2400"]
        """
        self.instrs["2400"] = Keithley2400(address)

    def load_6221(self, address: str) -> None:
        """
        load Keithley 6221 instrument according to the address, store it in self.instrs["6221"]
        """
        self.instrs["6221"] = Keithley6221(address)
        self.setup_6221()

    def load_ITC503(self, gpib_up: str, gpib_down: str) -> None:
        """
        load ITC503 instruments according to the addresses, store them in self.instrs["itc503"] in corresponding order

        Args:
            addresses (List[str]): the addresses of the ITC503 instruments (take care of the order)
        """
        self.instrs["itc503"] = ITCs(gpib_up, gpib_down)

    def load_mercury_ips(self, address: str = "TCPIP0::10.101.30.192::7020::SOCKET", if_print: bool = False, limit_sphere: float = 14) -> None:
        """
        load Mercury iPS instrument according to the address, store it in self.instrs["mercury_ips"]

        Args:
            address (str): the address of the instrument
            if_print (bool): whether to print the snapshot of the instrument
            limit (float): the limit of the field
        """
        self.instrs["mercury_ips"] = OxfordMercuryiPS("mips", address)
        if if_print:
            self.instrs["mercury_ips"].print_readable_snapshot(update=True)
        def spherical_limit(x,y,z) -> bool:
            return np.sqrt(x**2 + y**2 + z**2) <= limit_sphere
        self.instrs["mercury_ips"].set_new_field_limits(spherical_limit)
    
    def ramp_magfield(self, field: Tuple[float], rate: Tuple[float] = (0.00333,)*3, wait: bool = True ) -> None:
        """
        ramp the magnetic field to the target value with the rate

        Args:
            field (Tuple[float]): the target field coor
            rate (float): the rate of the field change (T/s)
            wait (bool): whether to wait for the ramping to finish
        """
        mips = self.instrs["mercury_ips"]
        if max(rate)*60 > 0.2:
            raise ValueError("The rate is too high, the maximum rate is 0.2 T/min")
        mips.GRPX.field_ramp_rate(rate[0])
        mips.GRPY.field_ramp_rate(rate[1])
        mips.GRPZ.field_ramp_rate(rate[2])
        mips.x_target(field[0])
        mips.y_target(field[1])
        mips.z_target(field[2])

        mips.ramp(mode = "simul")
        if wait:
            # the is_ramping() method is not working properly, so we use the following method to wait for the ramping to finish
            count = 0
            while count < 120:
                field_now = (mips.x_measured(),mips.y_measured(),mips.z_measured())
                if np.linalg.norm(field_now) - np.linalg.norm(field) < 0.01:
                    count += 1
                else:
                    count = 0
                time.sleep(1)
            print("ramping finished")

    def load_mercury_ipc(self, address: str = "TCPIP0::10.101.30.192::7020::SOCKET") -> None:
        """
        load Mercury iPS instrument according to the address, store it in self.instrs["mercury_ips"]
        """
        self.instrs["mercury_itc"] = MercuryITC(address)
        print(self.instrs["mercury_itc"].modules)
     
    def setup_SR830(self) -> None:
        """
        setup the SR830 instruments using pre-stored setups here, this function will not fully reset the instruments, only overwrite the specific settings here, other settings will all be reserved
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

#    def setup_2182(self, channel: Literal[0,1,2] = 1) -> None:
#        """
#        setup the Keithley 2182 instruments, overwrite the specific settings here, other settings will all be reserved
#        currently initialized to measure voltage
#        """
#        source_2182 = self.instrs["2182"]
#        source_2182.active_channel = channel
#        source_2182.channel_function = "voltage"
#        if channel == 1:
#            source_2182.ch_1.setup_voltage()
#        if channel == 2:
#            source_2182.ch_2.setup_voltage()

    def setup_6221(self) -> None:
        """
        setup the Keithley 6221 instruments, overwrite the specific settings here, other settings will all be reserved. Note that the waveform will not begin here
        """
        source_6221 = self.instrs["6221"]
        source_6221.clear()
        source_6221.waveform_function = "sine"
        source_6221.waveform_amplitude = 0
        source_6221.waveform_offset = 0
        source_6221.waveform_ranging = "fixed"
        source_6221.source_auto_range = False
        source_6221.waveform_use_phasemarker = True
        source_6221.waveform_phasemarker_line = 3
        source_6221.waveform_duration_set_infinity()
        source_6221.output_low_grounded = True


    def measure_iv_2400(self, v_max:float = 1E-4, v_step:float = 1E-5, curr_compliance: float = 1E-6, mode: Literal["0-max-0","0--max-max-0"] = "0-max-0",*, test: bool = True, source: int = None, drain: int = None, temp: int = None, tmpfolder: str = None) -> None:
        """
        Measure the IV curve using Keithley 2400 to test the contacts. No data will be saved. (meter need to be loaded before calling this function)

        Args:
            v_max (float): the maximum voltage to apply
            v_step (float): the step of the voltage
            curr_compliance (float): the current compliance
            mode (Literal["0-max-0","0--max-max-0"]): the mode of the measurement
            test (bool): whether for testing use, if True, no data will be saved, and the parameters after will not be used
        """
        if not test:
            file_path = self.get_filepath("iv__2-terminal",v_max,v_step, source, drain, mode,temp, tmpfolder=tmpfolder)
            file_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Max Voltage: {v_max} V")
        print(f"Voltage Step: {v_step} V")
        print(f"Max Curr: {curr_compliance} A")
        print(f"Meter: {self.instrs['2400'].adapter}")
        measure_delay = 0.3 # [s]
        instr_2400 = self.instrs["2400"]
        instr_2400.apply_voltage(compliance_current=curr_compliance)
        instr_2400.source_voltage = 0
        instr_2400.enable_source()
        instr_2400.measure_current()
        tmp_df = pd.DataFrame(columns=["V","I"])

        v_array = np.arange(0, v_max, v_step)
        if mode == "0-max-0":
            v_array = np.concatenate((v_array, v_array[::-1]))
        elif mode == "0--max-max-0":
            v_array = np.concatenate((-v_array, -v_array[::-1], v_array, v_array[::-1]))
        tmp_df["V"] = v_array
        i_array = np.zeros_like(v_array)

        for ii,v in enumerate(v_array):
            instr_2400.ramp_to_voltage(v,steps=10,pause=0.02)
            time.sleep(measure_delay)
            i_array[ii] = instr_2400.current
        tmp_df["I"] = i_array
        fig, ax = DataPlot.init_canvas(1,1,10,6)
        ax.plot(tmp_df["V"],tmp_df["I"])
        if not test:
            tmp_df.to_csv(file_path,sep="\t",index=False)

            #ax[0].plot(v_array[:ii+1],i_array[:ii+1])
            #ax[1].plot(v_array[:ii+1],v_array[:ii+1]/i_array[:ii+1],label="V/I")
            #ax[1].plot(v_array[1:ii+1],np.diff(v_array[:ii+1])/np.diff(i_array[:ii+1]),label="dV/dI")
            #ax[1].legend()
            #plt.draw()
        instr_2400.shutdown()
            



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

        fig, ax = DataPlot.init_canvas(1,2,15,10)
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
                    tmp_df.to_csv(file_path,sep="\t" ,index=False)
        except KeyboardInterrupt:
            print("Measurement interrupted")
        finally:
            tmp_df.to_csv(file_path,sep="\t", index=False)
            meter1.sine_voltage = 0
                

    @print_help_if_needed
    def measure_nonlinear_SR830(self,measurename_all, *var_tuple, tmpfolder:str = None, source : Literal["sr830", "6221"], tempsource: Literal["itc503","mercury_itc"], delay:int = 15) -> None:
        """
        conduct the 1-pair nonlinear measurement using 2 SR830 meters and store the data in the corresponding file. Using first meter to measure 2w signal and also as the source if appoint SR830 as source. (meters need to be loaded before calling this function). When using Keithley 6221 current source, the max voltage is the compliance voltage and the resistance does not have specific meaning, just used for calculating the current.

        Args:
            measurename_all (str): the full name of the measurement
            var_tuple (Tuple): the variables of the measurement, use "-h" to see the available options
            tmpfolder (str): the temporary folder to store the data
            source (Literal["sr830", "6221"]): the source of the measurement
            tempsource (Literal["itc503","mercury_itc"]): the source of the temperature
        """
        file_path = self.get_filepath(measurename_all,*var_tuple, tmpfolder=tmpfolder)
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
        measure_delay = delay # [s]
        tmp_df = pd.DataFrame(columns=["curr","X_2w","Y_2w","R_2w","phi_2w","X_1w","Y_1w","R_1w","phi_1w","T"])
        out_range = False

        self.setup_SR830()
        meter_2w = self.instrs['sr830'][0]
        meter_1w = self.instrs['sr830'][1]
        meter_2w.harmonic = 2
        meter_1w.harmonic = 1

        fig, ax = DataPlot.init_canvas(1,2,15,10)
        fig.tight_layout()
        phi = [i.twinx() for i in ax]
        if source == "sr830":
            meter_1w.reference_source_trigger = "POS EDGE"
            meter_2w.reference_source_trigger = "SINE"
            meter_2w.reference_source = "Internal"
            meter_2w.frequency = freq
        elif source == "6221":
            # 6221 use half peak-to-peak voltage as amplitude
            amp *= np.sqrt(2)
            source_6221 = self.instrs["6221"]
            self.setup_6221()
            source_6221.source_compliance = amp[-1]+0.1
            source_6221.source_range = amp[-1]/resist/0.7
            print(f"Keithley 6221 source range is set to {amp[-1]/resist/0.7} A")
            source_6221.waveform_frequency = freq
            meter_1w.reference_source_trigger = "POS EDGE"
            meter_2w.reference_source_trigger = "POS EDGE"
        try:
            for i, v in enumerate(amp):
                if source == "sr830":
                    meter_2w.sine_voltage = v
                elif source == "6221":
                    source_6221.waveform_abort()
                    source_6221.waveform_amplitude = v/resist
                    source_6221.waveform_arm()
                    source_6221.waveform_start()
                time.sleep(measure_delay)
                if meter_1w.is_out_of_range():
                    out_range = True
                if meter_2w.is_out_of_range():
                    out_range = True
                list_2w = meter_2w.snap("X","Y","R","THETA")
                list_1w = meter_1w.snap("X","Y","R","THETA")
                if tempsource == "itc503":
                    temp = self.instrs[tempsource].temperatures["pot_low"]
                    if temp >= 6.9:
                        temp = self.instrs[tempsource].temperatures["pot_high"]
                if tempsource == "mercury_itc":
                    ##TODO##
                    pass
                if source == "sr830":
                    list_tot = [v/resist] + list_2w + list_1w + [temp]
                if source == "6221":
                    list_tot = [v/resist/np.sqrt(2)] + list_2w + list_1w + [temp]
                print(f"curr: {list_tot[0]*1E3:.4f} mA\t 2w: {list_tot[1:5]}\t 1w: {list_tot[5:9]}\t T: {list_tot[-1]}")
                tmp_df.loc[len(tmp_df)] = list_tot
                if i % 10 == 0:
                    tmp_df.to_csv(file_path,sep="\t", index=False)
            self.dfs["nonlinear"] = tmp_df.copy()
            # rename the columns for compatibility with the plotting function
            self.rename_columns("nonlinear", {"Y_2w":"V2w","X_1w":"V1w"})
            self.set_unit({"I":"uA","V":"uV"})
            self.df_plot_nonlinear(handlers=(ax[1],phi[1],ax[0],phi[0]))
            if out_range:
                print("out-range happened, rerun")
                #self.measure_nonlinear_SR830(measurename_all, *var_tuple, tmpfolder=tmpfolder , source=source, tempsource=tempsource)
            #ax[0].plot(tmp_df["curr"],tmp_df["Y_2w"],color="r",label="2w")
            #phi[0].plot(tmp_df["curr"],tmp_df["phi_2w"],color="c",alpha=0.5,label="2w")
            #ax[1].plot(tmp_df["curr"],tmp_df["R_1w"],color="b",label="1w")
            #phi[1].plot(tmp_df["curr"],tmp_df["phi_1w"],color="m",alpha=0.5,label="1w")
        except KeyboardInterrupt:
            print("Measurement interrupted")
        finally:
            tmp_df.to_csv(file_path, sep="\t", index=False)
            if source == "sr830":
                meter_2w.sine_voltage = 0
            if source == "6221":
                source_6221.shutdown()

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
        with open(file_path, "w", encoding="utf-8") as f:
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

    def ramp_to_temperature(self, itc_name: Literal["up","down"], temp, P=None, I=None, D=None):
        """
        used to ramp the temperature of the ITCs, this method will not occupy the ITC if wait is False(default)
        """
        self.control_mode = ("RU",itc_name)
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
    def control_mode(self,mode: Tuple[Literal["LU","RU","LL","RL"],Literal["all","up","down"]]):
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
    def heater_gas_mode(self,mode: Tuple[Literal["MANUAL","AM","MA","AUTO"],Literal["all","up","down"]]):
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

    def set_pid(self, P:float, I:float, D:float, mode:Literal["all","up","down"]="all"):
        """ Sets the PID of the ITC503. A three-element list is required. The second elecment is "all" or "up" or "down" to specify which ITC503 to set. 
        The P,I,D here are the proportional band (K), integral action time (min), and derivative action time(min), respectively.
        """
        self.control_mode = ("RU",mode)
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
