# preloading 
- (make sure to set the environmental variables first)


```python
import os
from pathlib import Path
import datetime
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import functools
import itertools

import pylab_dk
import pylab_dk.pltconfig.color_preset as colors
from pylab_dk.constants import cm_to_inch, next_lst_gen, constant_generator
from pylab_dk.file_organizer import FileOrganizer
from pylab_dk.data_process import DataProcess
from pylab_dk.data_plot import DataPlot
# MeasureManager module will need the NIDAQmx module to work,
from pylab_dk.measure_manager import MeasureManager
```


```python
project_name = "Date-Material"
folder = FileOrganizer(project_name)
#Folder.add_measurement("RT")
#Folder.add_plan("RT","__whole, measure the whole RT")
```


```python
# to list the folder structure using tree method
folder.tree()
```

### Add measurement type if needed


```python
#FileOrganizer.add_measurement_type("V_source_sweep_ac","Vmax{maxv}V-step{stepv}V-freq{freq}Hz-{vhigh}-{vlow}", overwrite=False)
```


```python
FileOrganizer.name_fstr_gen("I_source_sweep_dc","V_sense","T_vary")
```




    ('I-V-T',
     'Imax{maxi}A-step{stepi}A-{iin}-{iout}-swpmode{mode}_V{note}-{vhigh}-{vlow}_Temp{Tstart}-{Tstop}K')




```python
#DataPlot.gui_pan_color()
```

# Setup and Parameter


```python
measurement = MeasureManager(project_name)
#measurement.get_visa_resources()   # list all VISA resources
#measurement.load_meter("6221","GPIB0::12::INSTR")
#measurement.load_meter("2182","GPIB0::7::INSTR")
#measurement.load_meter("2400","GPIB0::23::INSTR")
#measurement.load_meter("6430","GPIB0::24::INSTR")
#measurement.load_meter("sr830","GPIB0::8::INSTR","GPIB0::9::INSTR")
#measurement.load_mercury_ips("TCPIP0::0.0.0.0::7020::SOCKET")
#measurement.load_mercury_itc("TCPIP0::0.0.0.0::7020::SOCKET")
#measurement.load_ITC503("GPIB0::23::INSTR","GPIB0::24::INSTR")
#measurement.load_rotator()
```

*****
# Guidance for measurement

### *sweep modes:*
- 0-max-0
- 0--max-max-0
- manual

### Column Names:
- time (default in generator)
- X_source
- X (sense->ext) 
    - (if lock-in, then X, Y, R, Theta)

### Combination of Varies or Mappings
- set `if_combine_gen=False` in `get_mea_dict` to get the list of generators instead of a whole list generator
- use `constants.next_lst_gen` to get the next values list
- replace mapping generators with constant generators (careful about the values) to temporarily pause the mapping and do varying, and remember to vary circularly (hysteretically) use `reverse=True`

### About Latency
latency of python itself and GPIB connection can be ignored unless many meters are involved.
Latency for meters is listed below:
- 2182a Meter: ~0.4s
- 6221 SourceMeter dc: negligible


```python
# used for getting namestr for filling the vars_tup
FileOrganizer.name_fstr_gen("I_source_fixed_ac","V_sense", "T_vary")
```




    ('I-V-T',
     'Ifix{fixi}A-freq{freq}Hz-{iin}-{iout}_V{note}-{vhigh}-{vlow}_Temp{Tstart}-{Tstop}K')



# Common Head


```python
##TODO::##
# Parameters
step_time = 1 # s, wait time between each measurement step, note the delay time of python itself, especially during varying process
mapping = False
constrained = False  # constrained mapping means single sweep with multiple vars constrained by an equation (so just concatenation)
# NOTE the wait interval must be SHORTER than the ACTUAL step time x vary_criteria steps ()
wait_before_vary = 7 # s, wait time before starting the varying process
vary_criteria = 10  # the criteria step number for judging the end of varying process
vary_loop = True  # if scan hysteresis loop for varying

assert wait_before_vary < step_time * vary_criteria, "wait_before_vary must be shorter than the actual step time x vary_criteria steps"  # or the waiting will be misjudged as stability
if vary_loop:
    vary_criteria *= 3  # avoid misjudging at the turning point

# setup related meters IF SPECIAL PARAMETERS ARE NEEDED
#measurement.instrs["6221"][0].setup(low_grounded=False)

# setup mapping IF NEEDED
if mapping:
    def map_func(x):
        """
        the correspondance between mapping variables, default is equal
        """
        return x

    # more mapping variables can be included
    # note the order ((1,2) x (3,4) -> 1,3; 1,4; 2,3; 2,4)
    map1_lst = np.concatenate([np.arange(-10, -5, 0.2),np.arange(-5, 5, 0.1),np.arange(5, 10.001, 0.2)]) # achieve flexible mapping instead of evenly spaced

    map2_lst = map_func(map1_lst)  # or directly assign the mapping variables
    if not constrained:
        map_lsts = measurement.create_mapping(map1_lst, map2_lst, idxs=(0,1))
    else:
        map_lsts = [map1_lst, map2_lst]
else:
    map_lsts = None

# Core configuration
# Generate the measurement generator
mea_dict = measurement.get_measure_dict(
    mea_mods := ("V_source_sweep_dc","I_sense"),
    *(vars_tup := (1E-3, 5E-5, 1, 1, "0-max-0", "",1,1)), # refer to the last cell for the meaning of each element
    wrapper_lst=[measurement.instrs["6221"][0],
             measurement.instrs["2182"][0]],
    compliance_lst=["10V"],
    sr830_current_resistor = None, # only float
    if_combine_gen = True, # False for coexistence of vary and mapping
    special_name = None,
    sweep_tables = map_lsts,
    vary_criteria = vary_criteria
    )
##::TODO##

print(f"filepath: {mea_dict['file_path']}")
print(f"no of columns(with time column): {mea_dict['record_num']}")
print(f"vary modules: {mea_dict["vary_mod"]}")
```

# Normal Single Sweep


```python
# modify the plot configuration
# note i[0] is timer
measurement.live_plot_init(1,2,1)
measurement.start_saving(mea_dict["plot_record_path"],30)
for i in mea_dict["gen_lst"]:
    measurement.record_update(mea_dict["file_path"], mea_dict["record_num"], i)
    measurement.live_plot_update([0,0],
                                 [0,1],
                                 [0,0],
                                 [i[1],i[0]],
                                 [i[2],i[1]], incremental=True)
    time.sleep(step_time)
measurement.stop_saving()
```


```python
measurement.plot_df_cols(mea_mods, *vars_tup)
```

# Vary 
(recommended that only 1 vary is applied)


```python
# integrate vary functions and current value getters
vary_lst = []
curr_val_lst = []
set_val_lst = []
for i in mea_dict["vary_mod"]:
    match i:
        case "T":
            vary_lst.append(mea_dict["tmp_vary"][0])
            curr_val_lst.append(mea_dict["tmp_vary"][1])
            set_val_lst.append(mea_dict["tmp_vary"][2])
        case "B":
            vary_lst.append(mea_dict["mag_vary"][0])
            curr_val_lst.append(mea_dict["mag_vary"][1])
            set_val_lst.append(mea_dict["mag_vary"][2])
        case "Theta":
            vary_lst.append(mea_dict["angle_vary"][0])
            curr_val_lst.append(mea_dict["angle_vary"][1])
            set_val_lst.append(mea_dict["angle_vary"][2])

assert len(curr_val_lst) == 1, "vary_lst and curr_val_lst have lengths as 1"
assert len(vary_lst) == 1, "only one varying parameter is allowed"
```


```python
# modify the plot configuration
##TODO::##
measurement.live_plot_init(1,1,1)
##::TODO##
measurement.start_saving(mea_dict["plot_record_path"],30)
begin_vary = False
counter = 0
counter_vary = [0] * len(vary_lst)
for gen_i in mea_dict["gen_lst"]:
    measurement.record_update(mea_dict["file_path"], mea_dict["record_num"], gen_i)
    ##TODO:: add vary observing plotting##
    measurement.live_plot_update(0,0,0,gen_i[1],gen_i[2], incremental=True)
    ##::TODO##
    time.sleep(step_time)

    if not begin_vary:
        if counter >= wait_before_vary:
            for funci in vary_lst:
                funci()
            begin_vary = True
        else:
            counter += step_time

    if vary_loop:
        for idx, (i,j,k) in enumerate(zip(curr_val_lst, set_val_lst, vary_lst)):
            if abs(i() - j()) < i()/100:
                counter_vary[idx] += 1
                if counter_vary[idx] >= vary_criteria:
                    # NOTE: will cause delay among multiple varies (if any)
                    k(reverse=True)
                    counter_vary[idx] = 0
measurement.stop_saving()
```


```python
measurement.plot_df_cols(mea_mods, *vars_tup)
```

# Mapping


```python
# modify the plot configuration
measurement.live_plot_init(3,1,1,plot_types=[["scatter"],["scatter"],["contour"]])
measurement.start_saving(mea_dict["plot_record_path"],30)
for i in mea_dict["gen_lst"]:
    measurement.record_update(mea_dict["file_path"], mea_dict["record_num"], i)
    measurement.live_plot_update([0,1,2],[0]*3,[0]*3,
                                 [i[1],i[1],i[0]],
                                 [i[2],i[3],i[1]],
                                 [i[2]], incremental=True)
    time.sleep(step_time)
measurement.stop_saving()
```

# Special Measurement with Head
(combination of Vary (one) and Mapping)
(the combination could also be achieved by manually appointing Mapping (fixed python loop))


```python
##TODO::##
# Parameters
step_time = 1 # s, wait time between each measurement step, note the delay time of python itself, especially during varying process
mapping = False  # sweeping, not mapping
constrained = False  # constrained mapping means sweep on a curve (single sweep with multiple vars constrained by an equation) (so just concatenation)
# NOTE the wait interval must be SHORTER than the ACTUAL step time x vary_criteria steps ()
wait_before_vary = 7 # s, wait time before starting the varying process
vary_criteria = 10  # the criteria step number for judging the end of varying process
vary_loop = True  # if scan hysteresis loop for varying

assert wait_before_vary < step_time * vary_criteria, "wait_before_vary must be shorter than the actual step time x vary_criteria steps"  # or the waiting will be misjudged as stability
if vary_loop:
    vary_criteria *= 3  # avoid misjudging at the turning point

# setup related meters IF SPECIAL PARAMETERS ARE NEEDED
#measurement.instrs["6221"][0].setup(low_grounded=False)

# setup mapping IF NEEDED
if mapping:
    def map_func(x):
        """
        the correspondance between mapping variables, default is equal
        """
        return x

    # more mapping variables can be included
    # note the order ((1,2) x (3,4) -> 1,3; 1,4; 2,3; 2,4)
    map1_lst = np.concatenate([np.arange(-10, -5, 0.2),np.arange(-5, 5, 0.1),np.arange(5, 10.001, 0.2)]) # achieve flexible mapping instead of evenly spaced

    map2_lst = map_func(map1_lst)  # or directly assign the mapping variables
    if not constrained:
        map_lsts = measurement.create_mapping(map1_lst, map2_lst, idxs=(0,1))
    else:
        map_lsts = [map1_lst, map2_lst]
else:
    map_lsts = None

# Core configuration
# Generate the measurement generator
mea_dict = measurement.get_measure_dict(
    mea_mods := ("V_source_sweep_dc","I_sense","B_vary"),
    *(vars_tup := (1E-3, 5E-5, 1, 1, "0-max-0", "",1,1, vary_start := -1, vary_stop:= 1)), # refer to the last cell for the meaning of each element
    wrapper_lst=[measurement.instrs["6221"][0],
             measurement.instrs["2182"][0]],
    compliance_lst=["10V"],
    sr830_current_resistor = None, # only float
    if_combine_gen = False, # False for coexistence of vary and mapping
    special_name = None,
    sweep_tables = map_lsts,
    vary_criteria = vary_criteria
    )
##::TODO##

print(f"filepath: {mea_dict['file_path']}")
print(f"no of columns(with time column): {mea_dict['record_num']}")
print(f"vary modules: {mea_dict["vary_mod"]}")

# integrate vary functions and current value getters
vary_lst = []
curr_val_lst = []
set_val_lst = []
vary_bound_lst = []
for i in mea_dict["vary_mod"]:
    match i:
        case "T":
            vary_lst.append(mea_dict["tmp_vary"][0])
            curr_val_lst.append(mea_dict["tmp_vary"][1])
            set_val_lst.append(mea_dict["tmp_vary"][2])
            vary_bound_lst.append(mea_dict["tmp_vary"][3])
        case "B":
            vary_lst.append(mea_dict["mag_vary"][0])
            curr_val_lst.append(mea_dict["mag_vary"][1])
            set_val_lst.append(mea_dict["mag_vary"][2])
            vary_bound_lst.append(mea_dict["mag_vary"][3])
        case "Theta":
            vary_lst.append(mea_dict["angle_vary"][0])
            curr_val_lst.append(mea_dict["angle_vary"][1])
            set_val_lst.append(mea_dict["angle_vary"][2])
            vary_bound_lst.append(mea_dict["angle_vary"][3])

assert max(len(vary_lst), len(curr_val_lst), len(set_val_lst), len(vary_bound_lst)) == 1, "only one varying parameter is allowed"
assert min(len(vary_lst), len(curr_val_lst), len(set_val_lst), len(vary_bound_lst)) == 1, "only one varying parameter is allowed"
vary_start = vary_bound_lst[0][0]
vary_stop = vary_bound_lst[0][1]
```


```python
# modify the plot configuration
##TODO::##
measurement.live_plot_init(1,2,1)
##::TODO##
begin_vary = False
counter = 0
counter_vary = [0] * len(vary_lst)
# each row in list is a sweep step, so should be interruped each step
tmp_lst_swp = [[]] * len(mea_dict["swp_idx"])
plot_idx = 0
ori_path = mea_dict["plot_record_path"]
while True:
    measurement.start_saving(ori_path.parent / (ori_path.stem + f"{plot_idx}" + ori_path.suffix),30)
    measured_lst = next_lst_gen(mea_dict["gen_lst"])
    if measured_lst is None:
        break
    measurement.record_update(mea_dict["file_path"], mea_dict["record_num"], measured_lst)
    # substitute the swp to constant to do varies
    for n, i in enumerate(mea_dict["swp_idx"]):
        tmp_lst_swp[n] = mea_dict["gen_lst"][i]
        mea_dict["gen_lst"][i] = constant_generator(measured_lst[i])
    # begin a circular vary under each sweep step
    time.sleep(wait_before_vary)
    vary_lst[0]()
    counter = 0
    while counter < vary_criteria:
        measured_lst = next_lst_gen(mea_dict["gen_lst"])
        measurement.record_update(mea_dict["file_path"], mea_dict["record_num"], measured_lst)
        ##TODO:: add vary observing plotting##
        measurement.live_plot_update([0,0],[0,1],[0,0],
                                     [measured_lst[3], measured_lst[0]],[measured_lst[2], measured_lst[4]], incremental=True)

        if abs(curr_val_lst[0]() - set_val_lst[0]()) < 0.01:
            counter += 1
        else:
            counter = 0
        ##::TODO##
        time.sleep(step_time)

    if vary_loop:
        time.sleep(wait_before_vary)
        vary_lst[0](reverse=True)
        counter = 0
        while counter < vary_criteria:
            measured_lst = next_lst_gen(mea_dict["gen_lst"])
            measurement.record_update(mea_dict["file_path"], mea_dict["record_num"], measured_lst)
            ##TODO:: add vary observing plotting##
            measurement.live_plot_update([0,0],[0,1],[0,0],
                                         [measured_lst[3], measured_lst[0]],[measured_lst[2], measured_lst[4]], incremental=True)

            if abs(curr_val_lst[0]() - vary_stop) < 0.01:
                counter += 1
            else:
                counter = 0
            ##::TODO##
            time.sleep(step_time)
        measurement.live_plot_update(0,0,0,[],[])

    # substitute the swp to constant to do varies
    for n, i in enumerate(mea_dict["swp_idx"]):
        mea_dict["gen_lst"][i] = tmp_lst_swp[n]
        tmp_lst_swp[n] = []
    
    measurement.stop_saving()
    measurement.live_plot_update([0,0],[0,1],[0,0],[[],[]],[[],[]])
    plot_idx += 1
```


```python
measurement.plot_df_cols(mea_mods, *vars_tup)
```

# END of Template
******

# Contact Testing


```python
# Generate the measurement generator
mea_dict = measurement.get_measure_dict(
    mea_mods := ("I_source_sweep_dc","V_sense"),
    *(vars_tup := (5E-6, 1E-7, 0, 0, "0-max-0", "",0,0)), # refer to the last cell for the meaning of each element
    wrapper_lst=[measurement.instrs["6221"][0],
         measurement.instrs["2182"][0]],
    compliance_lst=["10V"],
    special_name = "contact")

# Parameters
step_time = 1 # s, wait time between each measurement step

print(f"filepath: {mea_dict['file_path']}")
print(f"no of columns(with time column): {mea_dict['record_num']}")
```


```python
# modify the plot configuration
measurement.live_plot_init(1,1,1)
measurement.start_saving(mea_dict["plot_record_path"],3)
for i in mea_dict["gen_lst"]:
    measurement.record_update(mea_dict["file_path"], mea_dict["record_num"], i)
    measurement.live_plot_update(0,0,0,i[0],i[1], incremental=True)
    time.sleep(step_time)
measurement.stop_saving()
```

# RT


```python
# Parameters
step_time = 1 # s, wait time between each measurement step
mapping = False
wait_before_vary = 30 # s, wait time before starting the varying process

# setup related meters IF SPECIAL PARAMETERS ARE NEEDED
#measurement.instrs["6221"][0].setup(low_grounded=False)

# setup mapping IF NEEDED
if mapping:
    ##TODO::##
    def map_func(x):
        """
        the correspondance between mapping variables, default is equal
        """
        return x

    # more mapping variables can be included
    # note the order ((1,2) x (3,4) -> 1,3; 1,4; 2,3; 2,4)
    map1_lst = np.concatenate([np.arange(-10, -5, 0.2),np.arange(-5, 5, 0.1),np.arange(5, 10.001, 0.2)]) # achieve flexible mapping instead of evenly spaced
    ##::TODO##

    map2_lst = map_func(map1_lst)
    map_lsts = measurement.create_mapping(map1_lst, map2_lst)
else:
    map_lsts = None

# Core configuration
# Generate the measurement generator
mea_dict = measurement.get_measure_dict(
    mea_mods := ("I_source_fixed_ac","V_sense", "V_sense", "T_vary"),
    *(vars_tup := ("1uA","13.671Hz",0,1, "0",0,1, "1",0,1, 300, 1.5)), # refer to the last cell for the meaning of each element
    wrapper_lst=[measurement.instrs["sr830"][0],
             measurement.instrs["sr830"][0],
             measurement.instrs["sr830"][1]],
    compliance_lst=["10V"],
    sr830_current_resistor = 1E5, # only float
    sweep_tables = map_lsts
    )
##::TODO##

print(f"filepath: {mea_dict['filepath']}")
print(f"no of columns(with time column): {mea_dict['record_num']}")
```


```python
vary_dict = {"temp_vary": False, "mag_vary": False, "angle_vary": False}
for i in ["temp_vary", "mag_vary", "angle_vary"]:
    if mea_dict[i] is not None:
        vary_dict[i] = True
        print(i)
```


```python
# modify the plot configuration
measurement.live_plot_init(1,2,1)
measurement.start_saving(mea_dict["plot_record_path"],30)
begin_vary = False
counter = 0
for i in mea_dict["gen_lst"]:
    measurement.record_update(mea_dict["file_path"], mea_dict["record_num"], i)
    measurement.live_plot_update(0,0,0,i[0],i[1], incremental=True)
    time.sleep(step_time)
    if not begin_vary:
        if counter >= wait_before_vary:
            for i in ["temp_vary", "mag_vary", "angle_vary"]:
                if vary_dict[i]:
                    mea_dict[i]()
            begin_vary = True
        else:
            counter += step_time
measurement.stop_saving()
```


```python
measurement.plot_df_cols(mea_mods, *vars_tup)
```

# Sweep


```python
##TODO::##
# Parameters
step_time = 1 # s, wait time between each measurement step
mapping = False

# setup related meters IF SPECIAL PARAMETERS ARE NEEDED
measurement.instrs["6221"][0].setup(low_grounded=False)

# setup mapping IF NEEDED
if mapping:
    ##TODO::##
    def map_func(x):
        """
        the correspondance between mapping variables, default is equal
        """
        return x

    # more mapping variables can be included
    # note the order ((1,2) x (3,4) -> 1,3; 1,4; 2,3; 2,4)
    map1_lst = np.concatenate([np.arange(-10, -5, 0.2),np.arange(-5, 5, 0.1),np.arange(5, 10.001, 0.2)]) # achieve flexible mapping instead of evenly spaced
    ##::TODO##

    map2_lst = map_func(map1_lst)
    map_lsts = measurement.create_mapping(map1_lst, map2_lst)
else:
    map_lsts = None

# Core configuration
# Generate the measurement generator
mea_dict = measurement.get_measure_dict(
    mea_mods := ("V_source_sweep_dc","I_sense"),
    *(vars_tup := (1E-3, 5E-5, 1, 1, "0-max-0", "",1,1)), # refer to the last cell for the meaning of each element
    wrapper_lst=[measurement.instrs["6221"][0],
             measurement.instrs["2182"][0]],
    compliance_lst=["10V"],
    sr830_current_resistor = None, # only float
    sweep_tables = map_lsts
    )
##::TODO##

print(f"filepath: {mea_dict['filepath']}")
print(f"no of columns(with time column): {mea_dict['record_num']}")
```


```python
# modify the plot configuration
measurement.live_plot_init(1,1,1)
measurement.start_saving(mea_dict["plot_record_path"],30)
for i in mea_dict["gen_lst"]:
    measurement.record_update(mea_dict["file_path"], mea_dict["record_num"], i)
    measurement.live_plot_update(0,0,0,i[0],i[1], incremental=True)
    time.sleep(step_time)
measurement.stop_saving()
```


```python
measurement.plot_df_cols(mea_mods, *vars_tup)
```

# Mapping


```python
##TODO::##
# Parameters
step_time = 1 # s, wait time between each measurement step
mapping = True
wait_before_vary = 30 # s, wait time before starting the varying process

# setup related meters IF SPECIAL PARAMETERS ARE NEEDED
#measurement.instrs["6221"][0].setup(low_grounded=False)

# setup mapping IF NEEDED
if mapping:
    ##TODO::##
    def map_func(x):
        """
        the correspondance between mapping variables, default is equal
        """
        return x

    # more mapping variables can be included
    # note the order ((1,2) x (3,4) -> 1,3; 1,4; 2,3; 2,4)
    map1_lst = np.concatenate([np.arange(-10, -5, 0.2),np.arange(-5, 5, 0.1),np.arange(5, 10.001, 0.2)]) # achieve flexible mapping instead of evenly spaced
    ##::TODO##

    map2_lst = map_func(map1_lst)
    map_lsts = measurement.create_mapping(map1_lst, map2_lst)
else:
    map_lsts = None

# Core configuration
# Generate the measurement generator
mea_dict = measurement.get_measure_dict(
    mea_mods := ("V_source_sweep_dc","I_sense"),
    *(vars_tup := (1E-3, 5E-5, 1, 1, "manual", "",1,1)), # refer to the last cell for the meaning of each element
    wrapper_lst=[measurement.instrs["6221"][0],
             measurement.instrs["2182"][0]],
    compliance_lst=["10V"],
    sr830_current_resistor = None, # only float
    special_name = "mapping",
    sweep_tables = map_lsts
    )
##::TODO##

print(f"filepath: {mea_dict['filepath']}")
print(f"no of columns(with time column): {mea_dict['record_num']}")
```


```python
# modify the plot configuration
measurement.live_plot_init(3,1,1,plot_types=[["scatter"],["scatter"],["contour"]])
measurement.start_saving(mea_dict["plot_record_path"],30)
for i in mea_dict["gen_lst"]:
    measurement.record_update(mea_dict["file_path"], mea_dict["record_num"], i)
    measurement.live_plot_update([0,1,2],[0]*3,[0]*3,
                                 [i[1],i[1],i[0]],
                                 [i[2],i[3],i[1]],
                                 [i[2]], incremental=True,
                                 max_points=None)
    time.sleep(step_time)
measurement.stop_saving()
```


```python

```
