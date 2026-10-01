from greengate.core import GreenGate, RouteResult
from greengate.measure import (ContendedMeasurement, Measurement,
                               ServiceLedger, measure, should_cascade)

__version__ = "0.3.0"
__all__ = [
    "GreenGate", "RouteResult",
    "measure", "Measurement", "ServiceLedger", "should_cascade",
    "ContendedMeasurement",
]


def __getattr__(name):
    # legacy demo class, loaded lazily to keep `import greengate` light
    if name == "GreenGateRouter":
        from greengate.router import GreenGateRouter
        return GreenGateRouter
    raise AttributeError(name)
