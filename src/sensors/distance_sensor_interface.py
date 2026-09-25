"""Abstract Base Class for distance sensors."""

from abc import ABC, abstractmethod

from ..core.models import SensorReading


class DistanceSensorInterface(ABC):
    """Interface defining operations for distance measuring hardware/simulators."""

    @abstractmethod
    def read_distance(self) -> SensorReading:
        """Reads current obstacle distance in meters.

        Returns:
            SensorReading data structure with distance and status.
        """
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        """Checks if distance sensor hardware/stream is responding properly.

        Returns:
            True if healthy, False otherwise.
        """
        pass
