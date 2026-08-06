import time
from dataclasses import dataclass, field

# Default carbon intensity: global average ~475 gCO2/kWh
DEFAULT_CARBON_INTENSITY = 475.0
# Power Usage Effectiveness for cloud data centres
DEFAULT_PUE = 1.2


@dataclass
class QueryCarbonRecord:
    """Carbon footprint record for a single query."""
    query: str
    model_used: str
    decision: str  # "ANSWER" or "ESCALATE"
    inference_time_s: float = 0.0
    energy_joules: float = 0.0
    carbon_grams: float = 0.0
    # Full accounting: if escalated, this records the wasted small-model cost
    wasted_energy_joules: float = 0.0
    wasted_carbon_grams: float = 0.0

    @property
    def total_carbon_grams(self) -> float:
        return self.carbon_grams + self.wasted_carbon_grams

    @property
    def total_energy_joules(self) -> float:
        return self.energy_joules + self.wasted_energy_joules


class CarbonProfiler:
    """Tracks energy and carbon cost per query.

    CPU mode: estimates energy from inference time and assumed TDP.
    GPU mode (future): will use pynvml for real power measurement.
    """

    def __init__(
        self,
        carbon_intensity: float = DEFAULT_CARBON_INTENSITY,
        pue: float = DEFAULT_PUE,
        cpu_tdp_watts: float = 15.0,
    ):
        self.carbon_intensity = carbon_intensity
        self.pue = pue
        self.cpu_tdp_watts = cpu_tdp_watts
        self.records: list[QueryCarbonRecord] = []
        self._timer_start: float = 0.0
        self._gpu_available = False

        self._try_init_gpu()

    def _try_init_gpu(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._gpu_available = True
        except Exception:
            self._gpu_available = False

    def start(self):
        if self._gpu_available:
            import pynvml
            self._power_before = pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle) / 1000.0
        self._timer_start = time.perf_counter()

    def stop(self) -> tuple[float, float]:
        """Stop timing and return (energy_joules, carbon_grams)."""
        elapsed = time.perf_counter() - self._timer_start

        if self._gpu_available:
            import pynvml
            power_after = pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle) / 1000.0
            avg_power = (self._power_before + power_after) / 2.0
        else:
            avg_power = self.cpu_tdp_watts * 0.4  # ~40% utilisation estimate

        energy_joules = avg_power * elapsed
        energy_kwh = energy_joules / 3_600_000.0  # J -> kWh
        carbon_grams = energy_kwh * self.pue * self.carbon_intensity

        return energy_joules, carbon_grams

    def record(
        self,
        query: str,
        model_used: str,
        decision: str,
        energy_joules: float,
        carbon_grams: float,
        wasted_energy: float = 0.0,
        wasted_carbon: float = 0.0,
    ) -> QueryCarbonRecord:
        rec = QueryCarbonRecord(
            query=query,
            model_used=model_used,
            decision=decision,
            inference_time_s=time.perf_counter() - self._timer_start,
            energy_joules=energy_joules,
            carbon_grams=carbon_grams,
            wasted_energy_joules=wasted_energy,
            wasted_carbon_grams=wasted_carbon,
        )
        self.records.append(rec)
        return rec

    def summary(self) -> dict:
        if not self.records:
            return {}
        total_carbon = sum(r.total_carbon_grams for r in self.records)
        total_energy = sum(r.total_energy_joules for r in self.records)
        total_wasted = sum(r.wasted_carbon_grams for r in self.records)
        answered = sum(1 for r in self.records if r.decision == "ANSWER")
        escalated = sum(1 for r in self.records if r.decision == "ESCALATE")
        return {
            "total_queries": len(self.records),
            "answered_locally": answered,
            "escalated": escalated,
            "escalation_rate": escalated / len(self.records),
            "total_energy_joules": total_energy,
            "total_carbon_grams": total_carbon,
            "wasted_carbon_grams": total_wasted,
            "mode": "gpu (pynvml)" if self._gpu_available else "cpu (estimated)",
        }
