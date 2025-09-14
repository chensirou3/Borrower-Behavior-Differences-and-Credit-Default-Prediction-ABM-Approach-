"""Main simulation loop and event generation."""

from acr.simulation.runner import simulate_events
from acr.simulation.schema import EVENT_SCHEMA

__all__ = ["simulate_events", "EVENT_SCHEMA"]
