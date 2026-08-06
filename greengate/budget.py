"""Sliding-window carbon budget (Chapter 3, Design Analysis 2.5.3).

Maintains a window of the last `window_s` seconds of routing decisions.
If the carbon spent inside the window would exceed budget K, escalations
are temporarily blocked (the query falls back to the small answer) until
older queries age out of the window.
"""

from collections import deque


class SlidingWindowBudget:
    def __init__(self, budget_g: float, window_s: float = 3600.0):
        self.budget_g = budget_g
        self.window_s = window_s
        self._events: deque[tuple[float, float]] = deque()  # (timestamp, carbon_g)

    def _expire(self, now: float):
        while self._events and self._events[0][0] < now - self.window_s:
            self._events.popleft()

    def window_carbon(self, now: float) -> float:
        self._expire(now)
        return sum(c for _, c in self._events)

    def allows(self, now: float, escalation_cost_g: float) -> bool:
        """Would escalating now keep the window under budget?"""
        return self.window_carbon(now) + escalation_cost_g <= self.budget_g

    def record(self, now: float, carbon_g: float):
        self._events.append((now, carbon_g))
