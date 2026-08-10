from greengate.core import GreenGate, RouteResult

__version__ = "0.1.0"
__all__ = ["GreenGate", "RouteResult"]


def __getattr__(name):
    # legacy demo class, loaded lazily to keep `import greengate` light
    if name == "GreenGateRouter":
        from greengate.router import GreenGateRouter
        return GreenGateRouter
    raise AttributeError(name)
