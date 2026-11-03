import re
import math
from typing import Literal
import numpy as np
from matplotlib import pyplot as plt
from .file_organizer import FileOrganizer


class Flakes:
    def __init__(self):
        self.dir_path = (
            FileOrganizer.load_third_party("flakes", location="out").parent / "flakes"
        )
        self.flakes_json = FileOrganizer.third_party_json
        self.coor_transition = {"sin": 0, "cos": 1, "x": 0, "y": 0}

    def list_flakes(self):
        """
        list all flakes labels in the json for further check or use
        """
        for i in self.flakes_json:
            print(i)

    def get_flake_info(self, label):
        """
        get the information of a flake label
        """
        if label not in self.flakes_json:
            print("flake not found")
        else:
            FileOrganizer.open_folder(self.dir_path / label)
            return self.flakes_json[label]

    def sync_flakes(self):
        """
        sync the flakes json with the local file
        """
        FileOrganizer.third_party_json = self.flakes_json
        FileOrganizer._sync_json("flakes")

    def add_flake(
        self, label: str, info: str, coor: tuple, *, ref1: tuple, ref2: tuple
    ):
        """
        add a new flake label with its information
        """
        self.flakes_json.update(
            {
                label: {
                    "info": info,
                    "ref_coor": coor,
                    "ref1_ref": ref1,
                    "ref2_ref": ref2,
                }
            }
        )
        self.sync_flakes()
        flake_dir = self.dir_path / label
        flake_dir.mkdir(exist_ok=True)
        FileOrganizer.open_folder(flake_dir)

    def del_flake(self, label):
        """
        delete a flake label
        """
        if label not in self.flakes_json:
            print("flake not found")
        else:
            del self.flakes_json[label]
            flake_dir = self.dir_path / label
            # no folders within this folder, only files
            for item in flake_dir.iterdir():
                item.unlink()
            flake_dir.rmdir()
            self.sync_flakes()

    def extract_flakes(self, label, *, ref1_new: tuple, ref2_new: tuple):
        """
        extract the flake label with new reference points
        """
        if label not in self.flakes_json:
            print("flake not found")
            return
        self.get_coor_transition(
            self.flakes_json[label]["ref1_ref"],
            ref1_new,
            self.flakes_json[label]["ref2_ref"],
            ref2_new,
        )
        self.transition_coors(self.flakes_json[label]["ref_coor"])

    def manual_calculator(self):
        """
        manually input a new flake label
        """
        p1_ref = input("first point reference coor(sep:/s):")
        p1_prac = input("first point practical coor(sep:/s):")
        p2_ref = input("second point reference coor(sep:/s):")
        p2_prac = input("second point practical coor(sep:/s):")

        vecp1_ref = list(map(float, re.split(" ", p1_ref)))
        vecp1_prac = list(map(float, re.split(" ", p1_prac)))
        vecp2_ref = list(map(float, re.split(" ", p2_ref)))
        vecp2_prac = list(map(float, re.split(" ", p2_prac)))

        self.get_coor_transition(vecp1_ref, vecp1_prac, vecp2_ref, vecp2_prac)

        while True:
            ref_in = input("coor in ref axes(sep:/s):")
            if ref_in == "":
                exit()
            vec_ref_in = list(map(float, re.split(" ", ref_in)))
            self.transition_coors(vec_ref_in)

    # the method for this method is calculating the relative transformation between two coordinate systems,
    # and then transform the coordinate in one system to another
    # different from the method later for the static method
    # thus two methods give opposite rotation (contravariant) and different translation
    def get_coor_transition(self, vecp1_ref, vecp1_prac, vecp2_ref, vecp2_prac):
        """
        calculate the transformation matrix and the displacement
        return the sin, cos of the rotation angle and the displacement

        Args:
            vecp1_ref: the first point in reference axes
            vecp1_prac: the first point in practical axes
            vecp2_ref: the second point in reference axes
            vecp2_prac: the second point in practical axes
        """
        # the transform matrix is solved analytically without approximation
        theta_sin = (
            (vecp2_prac[0] - vecp1_prac[0]) * (vecp2_ref[1] - vecp1_ref[1])
            - (vecp2_prac[1] - vecp1_prac[1]) * (vecp2_ref[0] - vecp1_ref[0])
        ) / ((vecp2_ref[1] - vecp1_ref[1]) ** 2 + (vecp2_ref[0] - vecp1_ref[0]) ** 2)

        if vecp1_ref[0] != vecp2_ref[0]:
            theta_cos = (
                (vecp2_prac[0] - vecp1_prac[0])
                - (vecp2_ref[1] - vecp1_ref[1]) * theta_sin
            ) / (vecp2_ref[0] - vecp1_ref[0])
        else:
            theta_cos = (vecp2_prac[1] - vecp1_prac[1]) / (vecp2_ref[1] - vecp1_ref[1])

        x = vecp2_prac[0] * theta_cos - vecp2_prac[1] * theta_sin - vecp2_ref[0]
        y = vecp2_prac[0] * theta_sin + vecp2_prac[1] * theta_cos - vecp2_ref[1]

        # the equation is over-constrained, so sin2+cos2 could be used as a indicator for numerical error
        print(f"sin2+cos2:{theta_sin**2 + theta_cos**2}")
        if theta_sin > 1:
            theta_sin = 1
        elif theta_sin < -1:
            theta_sin = -1
        print(
            f"rot_angle(only -90~90, x represents x & (-)180-x)\nangle:{math.asin(theta_sin) * 180 / math.pi}"
        )
        print(f"disp:({x},{y})")

        self.coor_transition.update(
            {"sin": theta_sin, "cos": theta_cos, "x": x, "y": y}
        )

    def transition_coors(self, vec_ref_in: tuple | list):
        """
        transform the coor in reference axes to practical axes
        """
        theta_sin, theta_cos, x, y = self.coor_transition.values()
        vec_out_x = (
            theta_cos * vec_ref_in[0]
            + theta_sin * vec_ref_in[1]
            + x * theta_cos
            + y * theta_sin
        )
        vec_out_y = (
            -theta_sin * vec_ref_in[0]
            + theta_cos * vec_ref_in[1]
            - x * theta_sin
            + y * theta_cos
        )
        print(f"coor in prac axes:{vec_out_x},{vec_out_y}")

    # write same static methods for calling without instantiation
    @staticmethod
    def plot_relative_pos(
        ref1: list | tuple,
        ref2: list | tuple,
        target: list | tuple,
        *,
        plot_handler: any = None,
    ) -> None:
        """
        plot the relative position of the target point to the reference points
        """
        relative_ref = np.array((ref2[0] - ref1[0], ref2[1] - ref1[1]))
        dist_ref = np.linalg.norm(relative_ref)
        # use special coordinate transformation to plot the relative position
        target_final = Flakes.coor_transition(
            ref1=ref1,
            ref1_new=(0, 0),
            ref2=ref2,
            ref2_new=(dist_ref, 0),
            target=target,
            suppress_print=True,
        )
        # plot two reference points on x-axis
        if plot_handler is None:
            plt.plot([0, dist_ref], [0, 0], "ro-")
            plt.scatter(target_final[0], target_final[1], c="purple", marker="x")
            plt.show()
        else:
            plot_handler.plot([0, dist_ref], [0, 0], "ro-")
            plot_handler.scatter(
                target_final[0], target_final[1], c="purple", marker="x"
            )

    # here use the transformation of object instead of coordinate system
    # tranlate the target point to the origin, rotate and then to the new location
    # check previous methods' comments for more details
    @staticmethod
    def coor_transition(
        *,
        ref1: list | tuple,
        ref1_new: list | tuple,
        ref2: list | tuple,
        ref2_new: list | tuple,
        target: list | tuple,
        suppress_print: Literal["plot"] | bool = False,
    ) -> tuple[float, float]:
        """
        calculate the transformation matrix and the displacement
        return the sin, cos of the rotation angle and the displacement

        Args:
            ref1: the first point in reference axes
            ref1_new: the first point in practical axes
            ref2: the second point in reference axes
            ref2_new: the second point in practical axes
            target: the target point in reference axes
            suppress_print: suppress the print and plot output (used for calling in other methods)
                            when set to "plot", only print info without plot
        """
        relative_ref = complex(ref2[0] - ref1[0], ref2[1] - ref1[1])
        relative_ref_new = complex(ref2_new[0] - ref1_new[0], ref2_new[1] - ref1_new[1])
        dist_ref = abs(relative_ref)
        dist_ref_new = abs(relative_ref_new)
        target_at_ori = complex(target[0] - ref1[0], target[1] - ref1[1])
        rot = (relative_ref_new / dist_ref_new) / (relative_ref / dist_ref)
        target_new = target_at_ori * rot + ref1_new[0] + 1j * ref1_new[1]

        if suppress_print in [False, "plot"]:
            print(
                f"magnitude(length) ratio(new/old):{abs(relative_ref_new / relative_ref) * 100} %"
            )
            print(f"rot_angle:{np.angle(rot) * 180 / np.pi}")
            print(f"disp:({ref1_new[0] - ref1[0]},{ref1_new[1] - ref1[1]})")
            print(f"coor in prac axes:{target_new.real},{target_new.imag}")
            if suppress_print != "plot":
                Flakes.plot_relative_pos(ref1, ref2, target)

        return target_new.real, target_new.imag
