from pyomnix.omnix_logger import get_logger
from .GenericMotorCLI_AdvancedMotor import GenericAdvancedMotorCLI

log = get_logger(__name__)


class IntegratedStepperMotor(GenericAdvancedMotorCLI):
    """
    This class is the common base class to all Integrated Stepper Controls.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CageRotator(IntegratedStepperMotor):
    """
    Represents a generic interface for Thorlabs Cage Rotator motor controllers.

    This class encapsulates the functionality provided by the Thorlabs .NET API for
    Cage Rotator motor controllers, integrating with the QCoDeS framework. It includes
    advanced motor control features and inherits from the GenericAdvancedMotorCLI.

    Args:
        polling_rate_ms: The polling rate in milliseconds for the stage status updates.
        api_interface: The API interface object for the stage. Optional
        actuator_name: The name of the actuator used in the stage. Optional
        *args: Variable positional arguments.
        **kwargs: Variable keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
