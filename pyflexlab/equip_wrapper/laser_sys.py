import time
from pyflexlab.drivers.Thorlabs.KDC101 import Thorlabs_KDC101
from pyflexlab.drivers.Thorlabs.Thorlabs_K10CR1_DotNet import ThorlabsK10CR1
from pyflexlab.drivers.Thorlabs.PM100D import Thorlab_PM100D
from pyflexlab.drivers.Thorlabs.Thorlabs_KSC101_DotNet import ThorlabsKSC101
from pyflexlab.drivers.nktlaser import NKTControl
from pyflexlab.drivers.shutter import Shutter
from pyflexlab.constants import LOCAL_DB_PATH
from pyomnix.omnix_logger import get_logger

logger = get_logger(__name__)


class LaserSys:
    def __init__(self):
        logger.info("Initializing LasorSys, the connection ports and the serial numbers are statically set in the code.")
        self.ports = {
            "PM100D": "USB0::0x1313::0x8078::P0032564::0::INSTR",
            "PM100D_2": "USB0::0x1313::0x8078::PM003618::0::INSTR",
            "FLU15": "COM6",
            "SC10": "COM1",
        }
        
        self.serial_no = {
            "ThorK10CR1" : 55455804, # rotatian of ND
            "ThorK10CR1_2" : 55378984, # rotatian of Polar 
            "KDC101PRM1" : "27005380", # rotatian of powermeter 
            "KSC101SH05" : "68800919",
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
        #self._apt_server = Thorlabs_APT(LOCAL_DB_PATH / "Thorlab" / "APT.dll")
        self.pm100d = Thorlab_PM100D("pm100d1", self.ports["PM100D"]) if self.switch_dict["PM100D"] else None
        self.pm100d_2 = Thorlab_PM100D("pm100d2", self.ports["PM100D_2"]) if self.switch_dict["PM100D_2"] else None
        # mot
        self.k10cr1 = ThorlabsK10CR1("k10cr11", self.serial_no["ThorK10CR1"], dll_directory=LOCAL_DB_PATH / "Thorlab") if self.switch_dict["ThorK10CR1"] else None
        # polar
        self.k10cr1_2 = ThorlabsK10CR1("k10cr12", self.serial_no["ThorK10CR1_2"], dll_directory=LOCAL_DB_PATH / "Thorlab") if self.switch_dict["ThorK10CR1_2"] else None
        self.kdc101 = Thorlabs_KDC101("kdc101", self.serial_no["KDC101PRM1"], dll_dir=LOCAL_DB_PATH / "Thorlab") if self.switch_dict["KDC101PRM1"] else None
        self.laser = NKTControl(self.ports["FLU15"]) if self.switch_dict["FLU15"] else None
        self.shutter = ThorlabsKSC101("shutter", self.serial_no["KSC101SH05"]) if self.switch_dict["KSC101SH05"] else None

    def _init_instruments(self):
        if self.k10cr1:
            self.k10cr1.enable()
        if self.k10cr1_2:
            self.k10cr1_2.enable()
        if self.shutter:
            self.shutter.operating_mode("Manual")

    def poweralign(self, wavelength, power):
        """
        Align the power to a certain value (in uW) at a certain wavelength (in nm)
        The powermeter used depends on the wavelength
        Use shutter instead of laser emission switch to protect the laser and allow for faster response
        """
        Pfactor = 1E6
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
            logger.raise_error(f"Invalid wavelength {wavelength} for power alignment", ValueError)

        # emission off first
        #self.laser.emission_off()
        self.shutter.operating_state("Inactive")
        time.sleep(2)
        outcell = _get_power()
        darkpower = outcell * Pfactor
        time.sleep(1)

        # max power
        #self.laser.emission_on()
        self.k10cr1.position(0)
        time.sleep(0.1)
        self.laser.set_power_level(45)
        time.sleep(2)
        self.shutter.operating_state("Active")
        time.sleep(2)
        outcell = _get_power()
        maxpower = outcell * Pfactor
        time.sleep(1)
        logger.info(f"Dark power: {darkpower:.3f}, max power: {maxpower:.1f}")

        # check power
        if abs(maxpower - darkpower) < power:
            logger.raise_error(f"Power not enough, maxpower = {maxpower}, darkpower = {darkpower}", ValueError)

        time.sleep(0.1)

        # ND Filter
        inP = 0
        #self.laser.set_power_level(inP)
        #time.sleep(0.5)
        outcell = _get_power()
        rPower = outcell * Pfactor - darkpower
        setPower = power
        while True:
            time.sleep(0.1)
            if rPower - setPower >= 50:
                inP = inP + 12
            elif rPower - setPower <= -50:
                inP = inP - 12
            elif 10 <= rPower - setPower < 50:
                inP = inP + 5
            elif -50 < rPower - setPower <= -10:
                inP = inP - 5
            elif 3 <= rPower - setPower < 10:
                inP = inP + 1
            elif -10 < rPower - setPower <= -3:
                inP = inP - 1
            elif 0.6 <= rPower - setPower < 3:
                inP = inP + 0.05
            elif -3 < rPower - setPower <= -0.6:
                inP = inP - 0.05
            elif -0.6 < rPower - setPower < 0.6:
                self.laser.set_power_level(inP)
                break
            else:
                logger.raise_error(f"Power alignment error: {rPower - setPower}", ValueError)

            time.sleep(0.2)
            self.k10cr1.position(inP)
            self.laser.set_power_level(inP)
            time.sleep(0.1)
            outcell = _get_power()
            rPower = outcell * Pfactor - darkpower
            time.sleep(0.5)
            logger.info(f"Current power: {rPower:.1f}")

        self.shutter.operating_state("Inactive")
        self.kdc101.go_home()
        logger.info("Power Aligned")

    def poweron(self):
        self.shutter.operating_state("Active")
        time.sleep(0.5)
        self.laser.emission_on()
        time.sleep(0.5)
        logger.info("Laser On")

    def poweroff(self):
        self.laser.emission_off()
        time.sleep(0.5)
        self.shutter.operating_state("Inactive")
        time.sleep(0.5)
        logger.info("Laser Off")