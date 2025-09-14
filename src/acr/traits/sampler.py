"""Trait sampling implementations."""

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from acr.config.schema import TraitsConfig, TraitDistConfig


class TraitSampler(Protocol):
    """Protocol for trait sampling."""
    
    def sample(self, N: int, rng: np.random.Generator) -> pd.DataFrame:
        """Sample traits for N individuals.
        
        Args:
            N: Number of individuals
            rng: Random number generator
            
        Returns:
            DataFrame with columns: gamma, beta, kappa, omega, eta
        """
        ...


class IndependentTraitSampler:
    """Independent truncated normal trait sampler (Stage 0).
    
    Samples each trait independently from truncated normal distributions
    based on the configuration parameters.
    """
    
    def __init__(self, config: TraitsConfig):
        """Initialize the sampler.
        
        Args:
            config: Traits configuration
        """
        self.config = config
    
    def sample(self, N: int, rng: np.random.Generator) -> pd.DataFrame:
        """Sample traits for N individuals.
        
        Args:
            N: Number of individuals
            rng: Random number generator
            
        Returns:
            DataFrame with columns: gamma, beta, kappa, omega, eta
        """
        traits = {}
        
        # Sample each trait independently
        for trait_name in ['gamma', 'beta', 'kappa', 'omega', 'eta']:
            trait_config = getattr(self.config, trait_name)
            traits[trait_name] = self._sample_trait(trait_config, N, rng)
        
        return pd.DataFrame(traits)
    
    def _sample_trait(
        self, 
        config: TraitDistConfig, 
        N: int, 
        rng: np.random.Generator
    ) -> np.ndarray:
        """Sample a single trait from truncated normal distribution.
        
        Args:
            config: Configuration for this trait
            N: Number of samples
            rng: Random number generator
            
        Returns:
            Array of sampled values
        """
        mean = config.mean
        sd = config.sd
        min_val = config.min
        max_val = config.max
        
        # Handle truncation bounds
        if min_val is None and max_val is None:
            # No truncation - use standard normal
            return rng.normal(mean, sd, N)
        
        # Convert to standard normal bounds for scipy.stats.truncnorm
        a = -np.inf if min_val is None else (min_val - mean) / sd
        b = np.inf if max_val is None else (max_val - mean) / sd
        
        # Sample from truncated normal
        samples = truncnorm.rvs(
            a=a, b=b, loc=mean, scale=sd, size=N, random_state=rng
        )
        
        return samples


# Placeholder classes for future stages
class MixtureTraitSampler:
    """Mixture trait sampler with prototypes (Stage 2).
    
    Samples from a mixture of conservative/mainstream/aggressive prototypes
    with added noise.
    """
    
    def __init__(self, config: TraitsConfig):
        self.config = config
        raise NotImplementedError("MixtureTraitSampler will be implemented in Stage 2")


class CopulaTraitSampler:
    """Copula-based trait sampler with correlations (Stage 2).
    
    Samples traits with specified correlation structure using copulas.
    """
    
    def __init__(self, config: TraitsConfig):
        self.config = config
        raise NotImplementedError("CopulaTraitSampler will be implemented in Stage 2")


class LHSTraitSampler:
    """Latin Hypercube Sampling trait sampler (Stage 2).
    
    Uses Latin Hypercube Sampling for better space coverage.
    """
    
    def __init__(self, config: TraitsConfig):
        self.config = config
        raise NotImplementedError("LHSTraitSampler will be implemented in Stage 2")
