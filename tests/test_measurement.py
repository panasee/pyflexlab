import pyflexlab
from prefect import flow
from pyflexlab.measure_flow import MeasureFlow
import numpy as np

@flow
def test_flow(measure_flow: MeasureFlow):
    measure_flow.measure_Vswp_I_vicurve_task(
        vmax=1,
        vstep=0.1,
        high=1,
        low=0,
        swpmode="0--max-max-0",
        meter=measure_flow.instrs["tests"][0],
        compliance=1e-12,
        folder_name="",
        step_time=0.5,
        individual_plot=True,
    )

    measure_flow.measure_VV_II_BTvary_rt_task(
        vds=1,
        ds_high=1,
        ds_low=0,
        ds_meter=measure_flow.instrs["tests"][0],
        ds_compliance=1e-12,
        vg=0,
        vg_high=1,
        vg_meter=measure_flow.instrs["tests"][1],
        vg_compliance=1e-12,
        field=0,
        temperature_start=2,
        temperature_end=100,
        folder_name="",
        step_time=0.7,
        wait_before_vary=13,
        vary_loop=False,
        individual_plot=True,
    )

    measure_flow.measure_VVswp_II_BT_gateswp_task(
        vds=1,
        ds_high=1,
        ds_low=0,
        ds_meter=measure_flow.instrs["tests"][0],
        ds_compliance=1e-12,
        vg_max=50,
        vg_step=0.1,
        vg_high=1,
        vg_swpmode="0--max-max-0",
        vg_swp_lst=None,
        vg_meter=measure_flow.instrs["tests"][1],
        vg_compliance=1e-12,
        field=0,
        temperature=0,
        folder_name="",
        step_time=0.5,
        individual_plot=True,
    )

    measure_flow.measure_VswpV_II_BT_vicurve_task(
        vds_max=1,
        vds_step=0.01,
        ds_high=1,
        ds_low=0,
        vds_swpmode="0--max-max-0",
        vds_swp_lst=None,
        ds_meter=measure_flow.instrs["tests"][0],
        ds_compliance=1e-12,
        vg=0,
        vg_high=1,
        vg_meter=measure_flow.instrs["tests"][1],
        vg_compliance=1e-12,
        field=0,
        temperature=0,
        folder_name="",
        step_time=0.3,
        individual_plot=True,
    )

    measure_flow.measure_VV_II_BvaryT_rhloop_task(
        vds=1,
        ds_high=1,
        ds_low=0,
        ds_meter=measure_flow.instrs["tests"][0],
        ds_compliance=1e-12,
        vg=0,
        vg_high=1,
        vg_meter=measure_flow.instrs["tests"][1],
        vg_compliance=1e-12,
        field_start=-7,
        field_end=7,
        temperature=0,
        folder_name="",
        step_time=0.3,
        wait_before_vary=5,
        vary_loop=True,
        individual_plot=True,
    )

    measure_flow.measure_VswpVswp_II_BT_dsgatemapping_task(
        constrained=False,
        vds_max=1,
        ds_map_lst=None,
        ds_high=1,
        ds_low=0,
        ds_meter=measure_flow.instrs["tests"][0],
        ds_compliance=1e-12,
        vg=0,
        gate_map_lst=None,
        vg_high=1,
        vg_meter=measure_flow.instrs["tests"][1],
        vg_compliance=1e-12,
        field=0,
        temperature=0,
        folder_name="",
        step_time=1,
        individual_plot=True,
        ds_gate_order=(0, 1),
    )

    measure_flow.measure_VswpVswp_II_BT_dsgatemapping_task(
        constrained=True,
        vds_max=1,
        ds_map_lst=None,
        ds_high=1,
        ds_low=0,
        ds_meter=measure_flow.instrs["tests"][0],
        ds_compliance=1e-12,
        vg=0,
        gate_map_lst=None,
        vg_high=1,
        vg_meter=measure_flow.instrs["tests"][1],
        vg_compliance=1e-12,
        field=0,
        temperature=0,
        folder_name="",
        step_time=1,
        individual_plot=True,
        ds_gate_order=(0, 1),
    )

    measure_flow.measure_VV_II_BTvary_rt_task(
        vds=1,
        ds_high=1,
        ds_low=0,
        ds_meter=measure_flow.instrs["tests"][0],
        ds_compliance=1e-12,
        vg=0,
        vg_high=1,
        vg_meter=measure_flow.instrs["tests"][1],
        vg_compliance=1e-12,
        field=0,
        temperature_start=2,
        temperature_end=100,
        folder_name="",
        step_time=0.7,
        wait_before_vary=13,
        vary_loop=False,
        individual_plot=True,
    )

    measure_flow.measure_VVswp_II_BT_gateswp_task(
        vds=1,
        ds_high=1,
        ds_low=0,
        ds_meter=measure_flow.instrs["tests"][0],
        ds_compliance=1e-12,
        vg_max=50,
        vg_step=0.1,
        vg_high=1,
        vg_swpmode="0--max-max-0",
        vg_swp_lst=None,
        vg_meter=measure_flow.instrs["tests"][1],
        vg_compliance=1e-12,
        field=0,
        temperature=0,
        folder_name="",
        step_time=0.5,
        individual_plot=True,
    )

    measure_flow.measure_VswpV_II_BT_vicurve_task(
        vds_max=1,
        vds_step=0.01,
        ds_high=1,
        ds_low=0,
        vds_swpmode="0--max-max-0",
        vds_swp_lst=None,
        ds_meter=measure_flow.instrs["tests"][0],
        ds_compliance=1e-12,
        vg=0,
        vg_high=1,
        vg_meter=measure_flow.instrs["tests"][1],
        vg_compliance=1e-12,
        field=0,
        temperature=0,
        folder_name="",
        step_time=0.3,
        individual_plot=True,
    )

    measure_flow.measure_VV_II_BvaryT_rhloop_task(
        vds=1,
        ds_high=1,
        ds_low=0,
        ds_meter=measure_flow.instrs["tests"][0],
        ds_compliance=1e-12,
        vg=0,
        vg_high=1,
        vg_meter=measure_flow.instrs["tests"][1],
        vg_compliance=1e-12,
        field_start=-7,
        field_end=7,
        temperature=0,
        folder_name="",
        step_time=0.3,
        wait_before_vary=5,
        vary_loop=True,
        individual_plot=True,
    )

    measure_flow.measure_VswpVswp_II_BT_dsgatemapping_task(
        constrained=False,
        vds_max=1,
        ds_map_lst=None,
        ds_high=1,
        ds_low=0,
        ds_meter=measure_flow.instrs["tests"][0],
        ds_compliance=1e-12,
        vg=0,
        gate_map_lst=None,
        vg_high=1,
        vg_meter=measure_flow.instrs["tests"][1],
        vg_compliance=1e-12,
        field=0,
        temperature=0,
        folder_name="",
        step_time=1,
        individual_plot=True,
        ds_gate_order=(0, 1),
    )

    measure_flow.measure_VswpVswp_II_BT_dsgatemapping_task(
        constrained=True,
        vds_max=1,
        ds_map_lst=None,
        ds_high=1,
        ds_low=0,
        ds_meter=measure_flow.instrs["tests"][0],
        ds_compliance=1e-12,
        vg=0,
        gate_map_lst=None,
        vg_high=1,
        vg_meter=measure_flow.instrs["tests"][1],
        vg_compliance=1e-12,
        field=0,
        temperature=0,
        folder_name="",
        step_time=1,
        individual_plot=True,
        ds_gate_order=(0, 1),
    )

def test_normal_func(measure_flow: MeasureFlow):
    measure_flow.measure_Vswp_I_vicurve(
        vmax=1,
        vstep=0.1,
        high=1,
        low=0,
        swpmode="0--max-max-0",
        meter=measure_flow.instrs["tests"][0],
        compliance=1e-12,
        folder_name="",
        step_time=0.3,
        individual_plot=True,
    )


if __name__ == "__main__":
    project_name = "Date-Material"
    measure_flow = MeasureFlow(project_name)
    measure_flow.load_meter("2450", "GPIB0::17::INSTR")
    #measure_flow.load_meter("tests", *[""]*3)
#    test_flow(measure_flow)

    #measure_flow.instrs["tests"][0].ramp_output("volt", -5)
    #measure_flow.instrs["tests"][1].ramp_output("volt", -50)
    measure_flow.measure_Vswp_I_vicurve(
        vmax=0.1,
        vstep=0.01,
        high=1,
        low=0,
        swpmode="0--max-max-0",
        meter=measure_flow.instrs["2450"][0],
        compliance=1e-1,
        folder_name="",
        step_time=0.2,
        individual_plot=True,
    )