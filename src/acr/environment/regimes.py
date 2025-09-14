"""Markov regime switching environment models (Stage 3+)."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class RegimeState:
    """A single regime state."""
    
    name: str
    mean_E: float      # Mean environment level
    volatility: float  # Volatility in this regime
    persistence: float # Probability of staying in regime


class MarkovRegimeSwitcher:
    """Markov regime switching environment model (Stage 3).
    
    Models environment as switching between different regimes
    (e.g., expansion, recession, crisis) with Markov transition probabilities.
    """
    
    def __init__(self, regimes: List[RegimeState], transition_matrix: np.ndarray):
        """Initialize regime switcher.
        
        Args:
            regimes: List of regime states
            transition_matrix: Transition probability matrix
        """
        self.regimes = regimes
        self.transition_matrix = transition_matrix
        
        if len(regimes) != transition_matrix.shape[0]:
            raise ValueError("Number of regimes must match transition matrix size")
        
        # Validate transition matrix
        if not np.allclose(transition_matrix.sum(axis=1), 1.0):
            raise ValueError("Transition matrix rows must sum to 1")
    
    def simulate(self, T: int, rng: np.random.Generator, initial_regime: int = 0) -> tuple:
        """Simulate regime switching environment.
        
        Args:
            T: Number of time periods
            rng: Random number generator
            initial_regime: Initial regime index
            
        Returns:
            Tuple of (E_t series, regime_sequence)
        """
        raise NotImplementedError("MarkovRegimeSwitcher will be implemented in Stage 3")


class ShockEnvironment:
    """Environment with one-time shocks (Stage 3).
    
    Adds discrete shocks to the base environment at specified times.
    """
    
    def __init__(self, shock_times: List[int], shock_magnitudes: List[float]):
        """Initialize shock environment.
        
        Args:
            shock_times: List of time periods when shocks occur
            shock_magnitudes: List of shock magnitudes
        """
        self.shock_times = shock_times
        self.shock_magnitudes = shock_magnitudes
        
        if len(shock_times) != len(shock_magnitudes):
            raise ValueError("Must have equal number of shock times and magnitudes")
    
    def apply_shocks(self, base_E_t: np.ndarray) -> np.ndarray:
        """Apply shocks to base environment series.
        
        Args:
            base_E_t: Base environment series
            
        Returns:
            Environment series with shocks applied
        """
        raise NotImplementedError("ShockEnvironment will be implemented in Stage 3")


# Predefined regime configurations for Stage 3
EXPANSION_REGIME = RegimeState(
    name="expansion",
    mean_E=-0.5,      # Loose monetary policy
    volatility=0.2,
    persistence=0.9
)

NORMAL_REGIME = RegimeState(
    name="normal", 
    mean_E=0.0,       # Neutral policy
    volatility=0.3,
    persistence=0.85
)

RECESSION_REGIME = RegimeState(
    name="recession",
    mean_E=0.8,       # Tight policy
    volatility=0.4,
    persistence=0.7
)

DEFAULT_REGIMES = [EXPANSION_REGIME, NORMAL_REGIME, RECESSION_REGIME]

# Example transition matrix (to be tuned in Stage 3)
DEFAULT_TRANSITION_MATRIX = np.array([
    [0.9,  0.08, 0.02],  # From expansion
    [0.1,  0.85, 0.05],  # From normal  
    [0.05, 0.25, 0.7 ]   # From recession
])
