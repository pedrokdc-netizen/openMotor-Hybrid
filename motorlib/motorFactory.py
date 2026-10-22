"""Construction helpers for serialized solid and hybrid motors."""

from .motor import Motor


def motorFromDict(data):
    """Build the correct motor implementation from serialized motor data."""
    motorType = data.get("motorType", "solid")
    if motorType == "solid":
        return Motor(data)
    if motorType == "hybrid":
        from .hybrid import HybridMotor

        return HybridMotor(data)
    raise ValueError("Unsupported motor type: {}".format(motorType))


def newMotor(motorType="solid"):
    """Create an empty motor of the requested type."""
    if motorType == "solid":
        return Motor()
    if motorType == "hybrid":
        from .hybrid import HybridMotor

        return HybridMotor()
    raise ValueError("Unsupported motor type: {}".format(motorType))
