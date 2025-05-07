import clr
import os
from pyflexlab.constants import LOCAL_DB_PATH
from pyomnix.omnix_logger import get_logger

from System import Enum  # Import Enum from the System namespace

logger = get_logger(__name__)


class Shutter:
    # Constants
    KINESISPATHDEFAULT = LOCAL_DB_PATH / "Thorlab"
    DEVICEMANAGERDLL = "Thorlabs.MotionControl.DeviceManagerCLI.dll"
    GENERICMOTORDLL = "Thorlabs.MotionControl.GenericMotorCLI.dll"
    SOLENOIDDLL = "Thorlabs.MotionControl.KCube.SolenoidCLI.dll"

    TPOLLING = 250  # Default polling time, in ms
    TIMEOUTSETTINGS = 5000  # Default timeout time for settings change

    def __init__(self, serialno: str | None = None):
        """Constructor - Instantiate shutter object"""
        self.load_dlls()  # Load DLLs (if not already loaded)
        if serialno is None:
            serial_numbers = self.list_devices()  # Build device list

            if not serial_numbers:
                raise Exception("No compatible Thorlabs K-Cube devices found!")

        # Initialize properties
        self.serialnumber = serialno
        self.controllername = None
        self.controllerdescription = None
        self.stagename = None

        # .NET objects
        self.deviceNET = None
        self.shutterSettingsNET = None
        self.currentDeviceSettingsNET = None
        self.deviceInfoNET = None
        self.initialized = False

        # Operating states and modes will be set during connection
        self.OPSTATE = None
        self.OPSTATE_ACTIVE = None
        self.OPSTATE_INACTIVE = None
        self.OPMODE = None
        self.OPMODE_MANUAL = None
        self.OPMODE_SINGLETOGGLE = None
        self.OPMODE_AUTOTOGGLE = None
        self.OPMODE_TRIGGERED = None

    def __del__(self):
        """Destructor - Clean up when object is deleted"""
        if self.deviceNET is not None:
            try:
                if self.isconnected:
                    self.operatingstate = "inactive"
                self.disconnect()
            except:
                pass

    # Properties with getters/setters
    @property
    def frontpanellock(self):
        self.deviceNET.RequestFrontPanelLocked()
        return self.deviceNET.GetFrontPanelLocked()

    @frontpanellock.setter
    def frontpanellock(self, lockstate):
        if not self.deviceNET.CanDeviceLockFrontPanel():
            print("The device does not support front panel locking.")
        else:
            if isinstance(lockstate, int):
                lockstate = bool(lockstate)
            if isinstance(lockstate, bool):
                self.deviceNET.SetFrontPanelLock(lockstate)

    @property
    def operatingmode(self):
        return str(self.deviceNET.GetOperatingMode())

    @operatingmode.setter
    def operatingmode(self, newmode):
        if isinstance(newmode, int):
            newmodeNET = self.OPMODE[newmode]
        elif isinstance(newmode, str):
            newmode = newmode.lower()
            if newmode == "manual":
                newmodeNET = self.OPMODE_MANUAL
            elif newmode == "singletoggle":
                newmodeNET = self.OPMODE_SINGLETOGGLE
            elif newmode == "autotoggle":
                newmodeNET = self.OPMODE_AUTOTOGGLE
            elif newmode == "triggered":
                newmodeNET = self.OPMODE_TRIGGERED
            else:
                raise ValueError("Operating mode not recognized!")
        self.deviceNET.SetOperatingMode(newmodeNET)

    @property
    def operatingstate(self):
        return str(self.deviceNET.GetOperatingState())

    @operatingstate.setter
    def operatingstate(self, newstate):
        if isinstance(newstate, int):
            newstateNET = self.OPSTATE[newstate]
        elif isinstance(newstate, str):
            newstate = newstate.lower()
            if newstate == "active":
                newstateNET = self.OPSTATE_ACTIVE
            elif newstate == "inactive":
                newstateNET = self.OPSTATE_INACTIVE
            else:
                raise ValueError("Operating state not recognized!")
        self.deviceNET.SetOperatingState(newstateNET)

    @property
    def state(self):
        return str(self.deviceNET.GetSolenoidState())

    @state.setter
    def state(self, value):
        raise AttributeError("You cannot set the State property directly!")

    @property
    def isconnected(self):
        return self.deviceNET.IsConnected

    @isconnected.setter
    def isconnected(self, value):
        raise AttributeError("You cannot set the IsConnected property directly!")

    # Device connection methods
    def connect(self, serialNo):
        """Connect to a shutter device with the given serial number"""
        if not self.initialized:
            if isinstance(serialNo, int):
                serialNo = str(serialNo)

            # Check if serial number corresponds to a KSC101
            device_prefix = "68"
            if serialNo.startswith(device_prefix):
                # Create the device instance
                kcube_solenoid = clr.GetClrType(
                    "Thorlabs.MotionControl.KCube.SolenoidCLI.KCubeSolenoid"
                )
                self.deviceNET = kcube_solenoid.CreateKCubeSolenoid(serialNo)
            else:
                raise Exception("Thorlabs Shutter and K-Cube not recognised")

            # Connect to the device
            self.deviceNET.Connect(serialNo)

            try:
                # Wait for settings to initialize
                if not self.deviceNET.IsSettingsInitialized:
                    self.deviceNET.WaitForSettingsInitialized(self.TIMEOUTSETTINGS)

                if not self.deviceNET.IsSettingsInitialized:
                    raise Exception(f"Unable to initialise device {serialNo}")

                # Start polling and enable device
                self.deviceNET.StartPolling(self.TPOLLING)
                self.deviceNET.EnableDevice()

                # Update device properties
                self.serialnumber = str(self.deviceNET.DeviceID)
                self.shutterSettingsNET = self.deviceNET.GetSolenoidConfiguration(
                    serialNo
                )
                self.stagename = str(self.shutterSettingsNET.DeviceSettingsName)
                self.currentDeviceSettingsNET = clr.GetClrType(
                    "Thorlabs.MotionControl.KCube.SolenoidCLI.ThorlabsKCubeSolenoidSettings"
                ).GetSettings(self.shutterSettingsNET)
                self.deviceInfoNET = self.deviceNET.GetDeviceInfo()
                self.controllername = str(self.deviceInfoNET.Name)
                self.controllerdescription = str(self.deviceInfoNET.Description)

                # Get operating states enum
                solenoid_status = clr.GetClrType(
                    "Thorlabs.MotionControl.KCube.SolenoidCLI.SolenoidStatus"
                )
                operating_states = solenoid_status.GetNestedType("OperatingStates")

                # Find enum values
                self.OPSTATE_INACTIVE = None
                self.OPSTATE_ACTIVE = None

                for value in Enum.GetValues(operating_states):
                    if "Inactive" in str(value):
                        self.OPSTATE_INACTIVE = value
                    elif "Active" in str(value):
                        self.OPSTATE_ACTIVE = value

                self.OPSTATE = [self.OPSTATE_INACTIVE, self.OPSTATE_ACTIVE]

                # Get operating modes enum
                operating_modes = solenoid_status.GetNestedType("OperatingModes")
                self.OPMODE_MANUAL = Enum.GetValues(operating_modes).GetValue(0)
                self.OPMODE_SINGLETOGGLE = Enum.GetValues(operating_modes).GetValue(1)
                self.OPMODE_AUTOTOGGLE = Enum.GetValues(operating_modes).GetValue(2)
                self.OPMODE_TRIGGERED = Enum.GetValues(operating_modes).GetValue(3)
                self.OPMODE = [
                    self.OPMODE_MANUAL,
                    self.OPMODE_SINGLETOGGLE,
                    self.OPMODE_AUTOTOGGLE,
                    self.OPMODE_TRIGGERED,
                ]

                self.initialized = True
                print(
                    f"Shutter {self.controllername} with S/N {self.serialnumber} is connected successfully!"
                )

            except Exception as e:
                raise Exception(f"Unable to initialise device {serialNo}: {str(e)}")
        else:
            raise Exception("Device is already connected.")

    def disconnect(self):
        """Disconnect from the shutter device"""
        if self.isconnected:
            try:
                self.deviceNET.StopPolling()
                self.deviceNET.DisableDevice()
                self.deviceNET.Disconnect()
                self.initialized = False
                print(
                    f"Shutter {self.controllername} with S/N {self.serialnumber} is disconnected successfully!"
                )
            except Exception as e:
                raise Exception(
                    f"Unable to disconnect device {self.serialnumber}: {str(e)}"
                )
        else:
            raise Exception("Device not connected.")

    def reset(self, serialNo):
        """Reset the device connection"""
        self.deviceNET.ResetConnection(serialNo)

    def status(self):
        """Get the device status"""
        self.deviceNET.RequestStatus()  # In principle excessive as polling is enabled
        res = self.deviceNET.GetStatusBits()
        reshexflip = format(res, "04x")[::-1]

        if res & 0x1:
            print("Solenoid output is enabled")
        else:
            print("Solenoid output is disabled")

        if reshexflip[3] == "2":
            print("Solenoid interlock state is enabled")
        else:
            print("Solenoid interlock state is disabled")

    # Static methods
    @staticmethod
    def list_devices():
        """List all connected shutter devices"""
        Shutter.load_dlls()

        device_manager = clr.GetClrType(
            "Thorlabs.MotionControl.DeviceManagerCLI.DeviceManagerCLI"
        )
        device_manager.BuildDeviceList()

        device_prefix = clr.GetClrType(
            "Thorlabs.MotionControl.KCube.SolenoidCLI.KCubeSolenoid"
        ).DevicePrefix
        serial_numbers_net = device_manager.GetDeviceList(device_prefix)

        # Convert .NET list to Python list
        serial_numbers = []
        for i in range(serial_numbers_net.Count):
            serial_numbers.append(str(serial_numbers_net[i]))

        return serial_numbers

    @staticmethod
    def load_dlls():
        """Load the required Thorlabs Kinesis DLLs"""
        try:
            # Check if DLLs are already loaded
            try:
                clr.GetClrType("Thorlabs.MotionControl.KCube.SolenoidCLI.KCubeSolenoid")
                return  # DLLs already loaded
            except:
                pass

            # Add DLL references
            clr.AddReference(
                os.path.join(Shutter.KINESISPATHDEFAULT, Shutter.DEVICEMANAGERDLL)
            )
            clr.AddReference(
                os.path.join(Shutter.KINESISPATHDEFAULT, Shutter.GENERICMOTORDLL)
            )
            clr.AddReference(
                os.path.join(Shutter.KINESISPATHDEFAULT, Shutter.SOLENOIDDLL)
            )

        except Exception as e:
            raise Exception(
                f"Unable to load .NET assemblies for Thorlabs Kinesis software: {str(e)}"
            )
