"""Measure energy and carbon for inference this library does not run itself.

The routing API in :mod:`greengate.core` owns both model tiers, which is the
right arrangement for a controlled comparison but the wrong one for a service
that already serves models through vLLM, TGI, Ollama or a provider SDK. The
helpers here separate the measurement instrument from the router so that an
existing pipeline can be profiled without being restructured:

    with greengate.measure(n_queries=1) as m:
        answer = my_pipeline(prompt)
    print(m.report())

Two honest limits apply, and the code enforces the first rather than hiding it.

Device power is sampled per GPU, so anything else running on the same GPU is
attributed to the measurement. :func:`measure` therefore records which compute
processes were present and reports the measurement as non-exclusive when it was
sharing the device; pass ``strict=True`` to raise instead. This is not a
theoretical concern: two evaluation runs launched concurrently during this
project's vision experiments inflated one tier's energy by 42%.

Under continuous batching many requests occupy the GPU at once and no
device-level meter can divide that energy between them. Rather than invent an
attribution, :class:`ServiceLedger` accounts at the level a service actually
cares about, totals over a window of work, and applies the break-even condition
to those totals.
"""

from __future__ import annotations

import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field

from greengate.profiler import (DEFAULT_CARBON_INTENSITY, DEFAULT_PUE,
                                CarbonProfiler)

__all__ = ["Measurement", "measure", "ServiceLedger", "should_cascade",
           "ContendedMeasurement"]


class ContendedMeasurement(RuntimeError):
    """Raised when a strict measurement shared its GPU with other work."""


def _compute_processes() -> list[int]:
    """PIDs with a compute context on any visible GPU, empty if unavailable."""
    try:
        import os

        import pynvml
        pynvml.nvmlInit()
        mine = os.getpid()
        pids = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                if p.pid != mine:
                    pids.append(p.pid)
        return sorted(set(pids))
    except Exception:
        return []


@dataclass
class Measurement:
    """Energy and carbon consumed inside a :func:`measure` block."""

    energy_joules: float
    carbon_grams: float
    duration_s: float
    avg_power_w: float
    n_queries: int
    mode: str
    exclusive: bool
    shared_with: list[int] = field(default_factory=list)
    label: str = ""

    @property
    def energy_per_query_j(self) -> float:
        return self.energy_joules / self.n_queries if self.n_queries else 0.0

    @property
    def carbon_per_query_g(self) -> float:
        return self.carbon_grams / self.n_queries if self.n_queries else 0.0

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "queries": self.n_queries,
            "duration_s": round(self.duration_s, 3),
            "avg_power_w": round(self.avg_power_w, 1),
            "energy_joules": round(self.energy_joules, 2),
            "carbon_grams": round(self.carbon_grams, 6),
            "energy_per_query_j": round(self.energy_per_query_j, 3),
            "carbon_per_query_g": round(self.carbon_per_query_g, 8),
            "measurement_mode": self.mode,
            "exclusive_gpu": self.exclusive,
            "shared_with_pids": self.shared_with,
        }

    def report(self) -> str:
        d = self.as_dict()
        head = f"{self.label or 'measurement'}: {d['queries']} quer" \
               f"{'y' if d['queries'] == 1 else 'ies'} in {d['duration_s']} s"
        body = (f"  {d['energy_joules']} J total, {d['energy_per_query_j']} J per query\n"
                f"  {d['carbon_grams']} g CO2 total, {d['carbon_per_query_g']} g per query\n"
                f"  {d['avg_power_w']} W average, measured by {d['measurement_mode']}")
        if not self.exclusive:
            body += (f"\n  WARNING: the GPU was shared with PIDs {self.shared_with} "
                     f"during this measurement, so the figures above include their "
                     f"consumption and overstate this workload.")
        return head + "\n" + body


@contextmanager
def measure(n_queries: int = 1, label: str = "", *, strict: bool = False,
            carbon_intensity: float = DEFAULT_CARBON_INTENSITY,
            pue: float = DEFAULT_PUE, poll_interval_s: float = 0.1):
    """Measure whatever runs inside the block.

    Args:
        n_queries: how many requests the block served, used for per-query
            figures. Leave at 1 to measure a single call, or pass the batch
            size to get an average.
        label: name for the measurement in the report.
        strict: raise :class:`ContendedMeasurement` if another process held a
            compute context on the GPU, instead of warning.
        carbon_intensity: grams of CO2 per kWh for the grid in use.
        pue: data-centre power usage effectiveness.

    Yields:
        A :class:`Measurement`, populated when the block exits.
    """
    profiler = CarbonProfiler(carbon_intensity=carbon_intensity, pue=pue,
                              poll_interval_s=poll_interval_s)
    before = _compute_processes()
    result = Measurement(0.0, 0.0, 0.0, 0.0, max(int(n_queries), 1),
                         "gpu (pynvml)" if profiler._gpu_available else "cpu (estimated)",
                         True, [], label)
    profiler.start()
    started = time.perf_counter()
    try:
        yield result
    finally:
        energy, carbon = profiler.stop()
        duration = time.perf_counter() - started
        after = _compute_processes()
        shared = sorted(set(before) | set(after))
        result.energy_joules = energy
        result.carbon_grams = carbon
        result.duration_s = duration
        result.avg_power_w = energy / duration if duration > 0 else 0.0
        result.exclusive = not shared
        result.shared_with = shared
        if shared:
            msg = (f"GreenGate measured while PIDs {shared} were also using the GPU; "
                   f"device-level power cannot be separated between processes, so "
                   f"these figures include their work.")
            if strict:
                raise ContendedMeasurement(msg)
            warnings.warn(msg, stacklevel=2)


def should_cascade(carbon_small: float, carbon_large: float,
                   escalation_rate: float) -> dict:
    """Apply the break-even condition to per-query costs.

    A cascade runs the small tier on every query and the large tier on the
    escalated fraction, so relative to always using the large tier the saving
    is ``S = 1 - C_s/C_l - e``. The cascade only pays while the escalation rate
    stays below ``1 - C_s/C_l``.

    Args:
        carbon_small: mean cost of one small-tier call, any consistent unit.
        carbon_large: mean cost of one large-tier call, same unit.
        escalation_rate: fraction of queries sent to the large tier, 0 to 1.

    Returns:
        The cost ratio, the break-even escalation rate, the remaining headroom,
        the predicted saving, and a plain verdict.
    """
    if carbon_large <= 0:
        raise ValueError("carbon_large must be positive")
    ratio = carbon_small / carbon_large
    break_even = 1.0 - ratio
    saving = 1.0 - ratio - escalation_rate
    if ratio >= 1.0:
        verdict = ("the small tier is not cheaper than the large tier, so no "
                   "escalation rate can save energy")
    elif saving > 0:
        verdict = (f"saves {saving * 100:.1f}% against always using the large tier; "
                   f"escalation may rise to {break_even * 100:.0f}% before that is lost")
    else:
        verdict = (f"costs {-saving * 100:.1f}% more than always using the large tier; "
                   f"escalation must fall below {break_even * 100:.0f}% to save")
    return {
        "cost_ratio_small_over_large": round(ratio, 4),
        "break_even_escalation_rate": round(break_even, 4),
        "headroom": round(break_even - escalation_rate, 4),
        "predicted_saving": round(saving, 4),
        "verdict": verdict,
    }


class ServiceLedger:
    """Aggregate accounting for a service whose requests are batched.

    Under continuous batching a device-level meter cannot attribute energy to
    individual requests, but it measures a window of work correctly. Record
    each window with the tier that served it and how many requests it covered,
    and the ledger derives mean per-query costs, the realised escalation rate
    and the break-even verdict from the totals.

        ledger = ServiceLedger()
        with greengate.measure(n_queries=128) as m:
            serve_batch_on_small_tier(...)
        ledger.add(m, tier="small")
        ...
        print(ledger.report())
    """

    def __init__(self):
        self.windows: list[dict] = []

    def add(self, measurement: Measurement | None = None, *, tier: str,
            queries: int | None = None, energy_joules: float | None = None,
            carbon_grams: float | None = None) -> None:
        """Record one measured window of work served by ``tier``.

        Either pass a :class:`Measurement`, or supply ``queries`` with
        ``energy_joules`` and/or ``carbon_grams`` directly.
        """
        if tier not in ("small", "large"):
            raise ValueError("tier must be 'small' or 'large'")
        if measurement is not None:
            queries = measurement.n_queries if queries is None else queries
            energy_joules = measurement.energy_joules
            carbon_grams = measurement.carbon_grams
            exclusive = measurement.exclusive
        else:
            exclusive = True
        if not queries:
            raise ValueError("queries must be a positive count")
        self.windows.append({
            "tier": tier,
            "queries": int(queries),
            "energy_joules": float(energy_joules or 0.0),
            "carbon_grams": float(carbon_grams or 0.0),
            "exclusive": exclusive,
        })

    def _totals(self, tier: str) -> tuple[int, float, float]:
        rows = [w for w in self.windows if w["tier"] == tier]
        return (sum(w["queries"] for w in rows),
                sum(w["energy_joules"] for w in rows),
                sum(w["carbon_grams"] for w in rows))

    def report(self) -> dict:
        """Totals, realised escalation rate and the break-even verdict."""
        n_small, e_small, c_small = self._totals("small")
        n_large, e_large, c_large = self._totals("large")
        if not n_small:
            return {"error": "no small-tier work recorded"}
        out = {
            "small_tier_queries": n_small,
            "large_tier_queries": n_large,
            "escalation_rate": round(n_large / n_small, 4) if n_small else 0.0,
            "total_energy_joules": round(e_small + e_large, 2),
            "total_carbon_grams": round(c_small + c_large, 6),
            "wasted_carbon_grams": round(
                c_small / n_small * n_large, 6) if n_small else 0.0,
            "contended_windows": sum(1 for w in self.windows if not w["exclusive"]),
        }
        if n_large:
            per_small = (c_small or e_small) / n_small
            per_large = (c_large or e_large) / n_large
            out.update(should_cascade(per_small, per_large, n_large / n_small))
        else:
            out["verdict"] = ("no escalations recorded; the service is running as a "
                              "static small tier")
        if out["contended_windows"]:
            out["warning"] = (f"{out['contended_windows']} window(s) shared the GPU with "
                              f"other processes and are overstated")
        return out
