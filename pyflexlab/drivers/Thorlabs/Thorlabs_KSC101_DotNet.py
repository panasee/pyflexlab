from typing import Optional
from pyomnix.omnix_logger import get_logger

from .private.DotNetAPI.KCube_SolenoidCLI import GenericKCubeSolenoidCLI
from .private.DotNetAPI.qcodes_thorlabs_integration import ThorlabsQcodesInstrument

logger = get_logger(__name__)


class ThorlabsKSC101(GenericKCubeSolenoidCLI, ThorlabsQcodesInstrument):
    """
    Driver for interfacing with the Thorlabs <model>
    via the QCoDeS framework and the .NET API.

    This class allows for control and management of the <model>.

    This class integrates the Thorlabs device, using the .NET API.
    For further .NET API details please refer to:
    .private/DotNetAPI/README.md

    Args:
        name (str): Name of the instrument.
        serial_number (str): The serial number of the Thorlabs device.
        startup_mode_value: .Net Enum value to be stored in '_startup_mode' and used as
                            'startupSettingsMode' in 'LoadMotorConfiguration'
            Valid startup modes:
                UseDeviceSettings: Use settings from device
                UseFileSettings: Use settings stored locally
                UseConfiguredSettings: Use one of the above according to chooice in
                                       Kinesis Sortware
        simulation (Optional[bool]): Flag to determine if the device is in simulation mode.
        polling_rate_ms (int): Polling rate in milliseconds for the device.
        dll_directory (Optional[str]): The directory where the DLL files are located.
    Raises:
        ValueError: If the model is not recognized as a known model.
    """

    def __init__(
        self,
        name: str,
        serial_number: str,
        startup_mode_value: str = "UseConfiguredSettings",  # UseDeviceSettings, UseFileSettings, UseConfiguredSettings
        simulation: Optional[bool] = False,
        polling_rate_ms: int = 250,
        dll_directory: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            name,  # Instrument (Qcodes)
            serial_number=serial_number,  # IGenericCoreDeviceCLI
            startup_mode_value=startup_mode_value,  # IGenericDeviceCLI
            simulation=simulation,  # ThorlabsQcodesInstrument
            polling_rate_ms=polling_rate_ms,  # IGenericDeviceCLI
            dll_directory=dll_directory,  # ThorlabsDLLMixin
            **kwargs,
        )

        if "(Simulated)" not in self.model():
            self.snapshot(True)

        self.connect_message()

        self.operating_mode = self.add_parameter(
            "operating_mode",
            get_cmd=self._get_operating_mode,
            set_cmd=self._set_operating_mode,
            val_mapping={
                "Manual": self._dll.SolenoidStatus.OperatingModes.Manual,
                "Single": self._dll.SolenoidStatus.OperatingModes.SingleToggle,
                "Auto": self._dll.SolenoidStatus.OperatingModes.AutoToggle,
                "Trigger": self._dll.SolenoidStatus.OperatingModes.Triggered,
            },
            docstring="Operating mode of the solenoid.",
        )

        self.operating_state = self.add_parameter(
            "operating_state",
            get_cmd=self._get_operating_state,
            set_cmd=self._set_operating_state,
            val_mapping={
                "Inactive": self._dll.SolenoidStatus.OperatingStates.Inactive,
                "Active": self._dll.SolenoidStatus.OperatingStates.Active,
                "Unknown": 0,
            },
            docstring="Operating state of the solenoid.",
        )

    def _import_device_dll(self):
        """Import the device-specific DLLs and classes from the .NET API."""
        self._add_dll("Thorlabs.MotionControl.GenericMotorCLI.dll")
        self._add_dll("Thorlabs.MotionControl.KCube.SolenoidCLI.dll")
        self._import_dll_class(
            "Thorlabs.MotionControl.KCube.SolenoidCLI", "KCubeSolenoid"
        )
        self._import_dll_class(
            "Thorlabs.MotionControl.KCube.SolenoidCLI", "SolenoidStatus"
        )

    def _get_api_interface_from_dll(self, serial_number: str):
        """Retrieve the API interface for the Thorlabs device using its serial number."""
        return self._dll.KCubeSolenoid.CreateKCubeSolenoid(
            serial_number
        )  # verify: <create_method>

    def _post_connection(self):
        """
        Will run after after establishing a connection, updating 'get_idn'
        and adding parameters 'model', 'serial_number' and 'firmware_version'.
        """
        knownmodels = [
            "KSC101",
        ]
        if self.model() not in knownmodels:
            raise ValueError(f"'{self.model}' is an unknown model.")

        # ensure the device settings have been initialized
        if not self._api_interface.IsSettingsInitialized():
            self._api_interface.WaitForSettingsInitialized(7000)
            if not self._api_interface.IsSettingsInitialized():
                raise Exception(f"Unable to initialise device {self._serial_number}")

    def _post_enable(self):
        """
        Will run after polling has started and the device/channel is enabled.
        """
        # Load any configuration settings needed by the device/channel
        serial = self._serial_number
        mode = self._startup_mode
        # self._configuration = self._api_interface.LoadMotorConfiguration(serial, mode)

    def _get_operating_mode(self) -> int:
        return self._api_interface.GetOperatingMode()

    def _set_operating_mode(self, value):
        self._api_interface.SetOperatingMode(value)

    def _get_operating_state(self) -> int:
        return self._api_interface.GetOperatingState()

    def _set_operating_state(self, value):
        self._api_interface.SetOperatingState(value)
