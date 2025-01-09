#!/usr/bin/env python
import time
import numpy as np
from pylab_dk.measure_manager import MeasureManager

project_name = "Date-Material"  # Name used only for test
measurement = MeasureManager(project_name)

# use random generator to generate data
# the generator will give string data to test the string and float conversion of the plotting function
def gen():
    for _ in range(50):
        yield [time.strftime("%H:%M:%S"), str(np.random.rand()), str(np.random.rand())]
test_lst = gen()

# modify the plot configuration
measurement.live_plot_init(2,2,1)
for i in test_lst:
    measurement.live_plot_update(0,0,0,i[1],i[2], incremental=True)
    measurement.live_plot_update(0,1,0,i[0],i[1], incremental=True)
    time.sleep(0.2)

# test all-at-once plotting
measurement.live_plot_update(1,0,0,[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9])