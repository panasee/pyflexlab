import time
from pyflexlab.drivers.Thorlabs.KDC101 import Thorlabs_KDC101
from pyflexlab.drivers.Thorlabs.Thorlabs_K10CR1_DotNet import ThorlabsK10CR1
from pyflexlab.drivers.Thorlabs.PM100D import Thorlab_PM100D
from pyflexlab.drivers.Thorlabs.Thorlabs_KSC101_DotNet import ThorlabsKSC101
from pyflexlab.drivers.nktlaser import NKTControl
from pyflexlab.constants import LOCAL_DB_PATH
from pyomnix.omnix_logger import get_logger
from pyomnix.utils.math import convert_unit

logger = get_logger(__name__)


class LaserSys:
    def __init__(self):
        logger.info(
            "Initializing LaserSys, the connection ports and the serial numbers are statically set in the code."
        )
        self.ports = {
            "PM100D": "USB0::0x1313::0x8078::P0032564::0::INSTR",
            "PM100D_2": "USB0::0x1313::0x8078::PM003618::0::INSTR",
            "FLU15": "COM6",
            "SC10": "COM1",
        }

        self.serial_no = {
            "ThorK10CR1": 55455804,  # rotatian of ND
            "ThorK10CR1_2": 55378984,  # rotatian of Polar
            "KDC101PRM1": "27005380",  # rotatian of powermeter
            "KSC101SH05": "68800919",
        }
        self.switch_dict = {
            "PM100D": True,
            "PM100D_2": True,
            "FLU15": True,
            "SC10": False,
            "ThorK10CR1": True,
            "ThorK10CR1_2": True,
            "KDC101PRM1": True,
            "KSC101SH05": True,
        }

        self._load_instruments()

    def _load_instruments(self):
        # self._apt_server = Thorlabs_APT(LOCAL_DB_PATH / "Thorlab" / "APT.dll")
        self.pm100d = (
            Thorlab_PM100D("pm100d1", self.ports["PM100D"])
            if self.switch_dict["PM100D"]
            else None
        )
        self.pm100d_2 = (
            Thorlab_PM100D("pm100d2", self.ports["PM100D_2"])
            if self.switch_dict["PM100D_2"]
            else None
        )
        # mot
        self.k10cr1 = (
            ThorlabsK10CR1(
                "k10cr11",
                self.serial_no["ThorK10CR1"],
                dll_directory=LOCAL_DB_PATH / "Thorlab",
            )
            if self.switch_dict["ThorK10CR1"]
            else None
        )
        # polar
        self.k10cr1_2 = (
            ThorlabsK10CR1(
                "k10cr12",
                self.serial_no["ThorK10CR1_2"],
                dll_directory=LOCAL_DB_PATH / "Thorlab",
            )
            if self.switch_dict["ThorK10CR1_2"]
            else None
        )
        self.kdc101 = (
            Thorlabs_KDC101(
                "kdc101",
                self.serial_no["KDC101PRM1"],
                dll_dir=LOCAL_DB_PATH / "Thorlab",
            )
            if self.switch_dict["KDC101PRM1"]
            else None
        )
        self.laser = (
            NKTControl(self.ports["FLU15"]) if self.switch_dict["FLU15"] else None
        )
        self.laser.connect()
        self.shutter = (
            ThorlabsKSC101("shutter", self.serial_no["KSC101SH05"])
            if self.switch_dict["KSC101SH05"]
            else None
        )

    def _init_instruments(self):
        if self.k10cr1:
            self.k10cr1.enable()
        if self.k10cr1_2:
            self.k10cr1_2.enable()
        if self.shutter:
            self.shutter.operating_mode("Manual")

    def power_align(
        self,
        wavelength: float | str,
        power: float | str,
        laser_change: bool = False,
    ) -> None:
        """
        USE SI UNIT
        Align the power to a certain value at a certain wavelength
        The powermeter used depends on the wavelength
        Use shutter instead of laser emission switch to protect the laser and allow for faster response
        """

        wavelength = convert_unit(wavelength, "nm")[0]
        power = convert_unit(power, "uW")[0]
        if 400 <= wavelength < 1100:
            self.pm100d_2.wavelength(wavelength)
            self.kdc101.move_to(60)

            def _get_power():
                return self.pm100d_2.power()
        elif 1100 <= wavelength < 1800:
            self.pm100d.wavelength(wavelength)
            self.kdc101.move_to(68)

            def _get_power():
                return self.pm100d.power()
        else:
            logger.raise_error(
                f"Invalid wavelength {wavelength} for power alignment", ValueError
            )

        # emission off first
        # self.laser_off()
        self.shutter_close()
        time.sleep(2)
        outcell = convert_unit(_get_power(), "uW")[0]
        darkpower = outcell
        time.sleep(1)

        # max power
        # self.laser.emission_on()
        self.k10cr1.position(0)
        time.sleep(0.1)
        #self.laser.set_power_level(100)
        time.sleep(2)
        self.shutter_open()
        time.sleep(2)
        outcell = convert_unit(_get_power(), "uW")[0]
        maxpower = outcell
        time.sleep(1)
        logger.info(f"Dark power: {darkpower:.3f}, max power: {maxpower:.1f}")

        # check power
        if abs(maxpower - darkpower) < power:
            logger.raise_error(
                f"Power not enough, maxpower = {maxpower}, darkpower = {darkpower}",
                ValueError,
            )

        time.sleep(0.1)

        # ND Filter
        inP = 0
        # self.laser.set_power_level(inP)
        # time.sleep(0.5)
        outcell = convert_unit(_get_power(), "uW")[0]
        rPower = outcell - darkpower
        setPower = power
        while True:
            time.sleep(0.1)
            if rPower - setPower >= 50:
                inP = inP + 17
            elif rPower - setPower <= -50:
                inP = inP - 17
            elif 10 <= rPower - setPower < 50:
                inP = inP + 7
            elif -50 < rPower - setPower <= -10:
                inP = inP - 7
            elif 3 <= rPower - setPower < 10:
                inP = inP + 2
            elif -10 < rPower - setPower <= -3:
                inP = inP - 2
            elif 0.6 <= rPower - setPower < 3:
                inP = inP + 0.3
            elif -3 < rPower - setPower <= -0.6:
                inP = inP - 0.3
            elif 0.1 <= rPower - setPower < 0.6:
                inP = inP + 0.1
            elif -0.6 < rPower - setPower <= -0.1:
                inP = inP - 0.1
            elif 0.03 <= rPower - setPower < 0.1:
                inP = inP + 0.03
            elif -0.1 < rPower - setPower <= -0.03:
                inP = inP - 0.03
            elif -0.03 < rPower - setPower < 0.03:
                break
            else:
                logger.raise_error(
                    f"Power alignment error: {rPower - setPower}", ValueError
                )

            time.sleep(0.2)
            logger.info(f"Setting ND filter to {inP}")
            self.k10cr1.position(inP)
            #self.laser.set_power_level(inP)
            time.sleep(0.1)
            outcell = convert_unit(_get_power(), "uW")[0]
            rPower = outcell - darkpower
            time.sleep(0.5)
            logger.info(f"Current power: {rPower:.1f}")

        self.shutter_close()
        self.kdc101.go_home()
        logger.info("Power Aligned")

    def laser_on(self):
        self.laser.emission_on()
        time.sleep(0.2)
        logger.info("Laser On")

    def laser_off(self):
        self.laser.emission_off()
        time.sleep(0.2)
        logger.info("Laser Off")

    def shutter_open(self):
        self.shutter.operating_state("Active")

    def shutter_close(self):
        self.shutter.operating_state("Inactive")
