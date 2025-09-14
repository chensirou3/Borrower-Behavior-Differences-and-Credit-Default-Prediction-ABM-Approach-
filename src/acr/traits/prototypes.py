"""Trait prototypes for mixture sampling (Stage 2+)."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class TraitPrototype:
    """A trait prototype with specific characteristics."""
    
    name: str
    gamma: float  # Risk appetite
    beta: float   # Financial discipline  
    kappa: float  # Behavioral volatility
    omega: float  # External shock sensitivity
    eta: float    # Learning/adaptation
    weight: float = 1.0  # Mixture weight


# Predefined prototypes for Stage 2
CONSERVATIVE_PROTOTYPE = TraitPrototype(
    name="conservative",
    gamma=1.5,   # Lower risk appetite
    beta=0.95,   # High financial discipline
    kappa=0.2,   # Low behavioral volatility
    omega=-0.3,  # Less sensitive to shocks
    eta=0.8,     # Good adaptation
    weight=0.3
)

MAINSTREAM_PROTOTYPE = TraitPrototype(
    name="mainstream", 
    gamma=2.0,   # Moderate risk appetite
    beta=0.85,   # Moderate financial discipline
    kappa=0.5,   # Moderate behavioral volatility
    omega=0.0,   # Average shock sensitivity
    eta=0.7,     # Average adaptation
    weight=0.5
)

AGGRESSIVE_PROTOTYPE = TraitPrototype(
    name="aggressive",
    gamma=2.8,   # High risk appetite
    beta=0.75,   # Lower financial discipline
    kappa=0.9,   # High behavioral volatility
    omega=0.4,   # More sensitive to shocks
    eta=0.6,     # Slower adaptation
    weight=0.2
)

DEFAULT_PROTOTYPES = [CONSERVATIVE_PROTOTYPE, MAINSTREAM_PROTOTYPE, AGGRESSIVE_PROTOTYPE]


def get_prototype_means(prototypes: List[TraitPrototype]) -> Dict[str, float]:
    """Get weighted mean values for each trait across prototypes.
    
    Args:
        prototypes: List of trait prototypes
        
    Returns:
        Dictionary of trait means
    """
    total_weight = sum(p.weight for p in prototypes)
    
    means = {}
    for trait in ['gamma', 'beta', 'kappa', 'omega', 'eta']:
        weighted_sum = sum(getattr(p, trait) * p.weight for p in prototypes)
        means[trait] = weighted_sum / total_weight
    
    return means


def sample_from_prototypes(
    prototypes: List[TraitPrototype],
    N: int,
    noise_sd: float,
    rng: np.random.Generator
) -> pd.DataFrame:
    """Sample traits from mixture of prototypes with noise.
    
    Args:
        prototypes: List of trait prototypes
        N: Number of samples
        noise_sd: Standard deviation of noise to add
        rng: Random number generator
        
    Returns:
        DataFrame with sampled traits
    """
    # Sample prototype assignments
    weights = np.array([p.weight for p in prototypes])
    weights = weights / weights.sum()
    
    prototype_indices = rng.choice(len(prototypes), size=N, p=weights)
    
    # Initialize output arrays
    traits = {trait: np.zeros(N) for trait in ['gamma', 'beta', 'kappa', 'omega', 'eta']}
    
    # Sample from each prototype
    for i, prototype in enumerate(prototypes):
        mask = prototype_indices == i
        n_samples = mask.sum()
        
        if n_samples == 0:
            continue
            
        for trait in traits.keys():
            base_value = getattr(prototype, trait)
            noise = rng.normal(0, noise_sd, n_samples)
            traits[trait][mask] = base_value + noise
    
    return pd.DataFrame(traits)
