"""
This file is to test the workflow of the measurement
"""

import numpy as np
from pyflexlab import MeasureFlow
from pyomnix import DataManipulator

testobj = MeasureFlow("Date-Material")
plotobj = DataManipulator(1)
testobj.load_fakes(5)
# the fake meters default to dc mode
testobj.instrs["fakes"][0].setup("ac")
testobj.instrs["fakes"][1].setup("ac")
map_lst1 = np.concatenate(
    [np.arange(0, 50, 5), np.arange(50, -50, -5), np.arange(-50, 0.1, 5)]
)
map_lst2 = map_lst1 / 10
map_lst3 = map_lst1 / 20
#testobj.measure_Vswp_V_vrcurve_lockin(
#    resistor = "5.1kOhm", 
#    vmax=1,
#    freq=13.317,
#    vstep=0.1,
#    high=0,
#    low=0,
#    swpmode="0-max-0",
#    meter=testobj.instrs["fakes"][0],
#    compliance=1e-3,
#    step_time=0.1,
#    plotobj=plotobj,
#    source_wait=0.5,
#)

testobj.measure_VV_V1wI_BTvary_rt_lockin(
    resistor=1e5,
    vds=1,
    freq=13.317,
    ds_high=0,
    ds_low=0,
    ds_meter=testobj.instrs["fakes"][0],
    ds_compliance=1e-3,
    vg=0,
    vg_high=0,
    vg_meter=testobj.instrs["fakes"][3],
    vg_compliance=1e-3,
    field=0,
    temperature_start=300,
    temperature_end=0,
    step_time=0.1,
    plotobj=plotobj,
)

#testobj.measure_VVswpVswp_VII_BT_dualgatemapping_ac(
#    constrained=True,
#    resistor=1e5,
#    vds=1,
#    freq=13.317,
#    ds_high=0,
#    ds_low=0,
#    ds_meter=testobj.instrs["fakes"][0],
#    ds_compliance=1e-3,
#    vg1_max=50,
#    vg1_map_lst=map_lst1,
#    vg1_high=0,
#    vg1_meter=testobj.instrs["fakes"][3],
#    vg1_compliance=5e-9,
#    vg2_max=5,
#    vg2_map_lst=map_lst1,
#    vg2_high=0,
#    vg2_meter=testobj.instrs["fakes"][4],
#    vg2_compliance=5e-9,
#    field=0,
#    temperature=300,
#    step_time=0.1,
#    plotobj=plotobj,
#)
#
#testobj.measure_VVswpVswp_III_BT_dualgatemapping(
#    constrained=True,
#    vds=1,
#    ds_high=0,
#    ds_low=0,
#    ds_meter=testobj.instrs["fakes"][2],
#    ds_compliance=1e-3,
#    vg1_max=50,
#    vg1_map_lst=map_lst1,
#    vg1_high=0,
#    vg1_meter=testobj.instrs["fakes"][3],
#    vg1_compliance=5e-9,
#    vg2_max=5,
#    vg2_map_lst=map_lst2,
#    vg2_high=0,
#    vg2_meter=testobj.instrs["fakes"][4],
#    vg2_compliance=5e-9,
#    field=0,
#    temperature=300,
#    plotobj=plotobj,
#)
