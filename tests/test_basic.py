"""Basic integration tests for ACR system."""

import numpy as np
import pandas as pd
import pytest

from acr.config.schema import Config
from acr.traits.sampler import IndependentTraitSampler
from acr.proxies.mapping import map_traits_to_proxies
from acr.environment.cycles import build_sine_env
from acr.dgp.default_risk import true_pd_vectorized, generate_defaults
from acr.simulation.runner import simulate_events


def test_config_creation():
    """Test that default configuration can be created."""
    config = Config()
    assert config.seed == 42
    assert config.population.N == 5000
    assert config.timeline.T == 120


def test_trait_sampling():
    """Test trait sampling functionality."""
    config = Config()
    sampler = IndependentTraitSampler(config.traits)
    rng = np.random.default_rng(42)
    
    traits = sampler.sample(100, rng)
    
    assert len(traits) == 100
    assert set(traits.columns) == {'gamma', 'beta', 'kappa', 'omega', 'eta'}
    
    # Check basic bounds
    assert traits['beta'].min() >= 0.60  # Min bound
    assert traits['beta'].max() <= 1.00  # Max bound
    assert traits['kappa'].min() >= 0.00  # Min bound


def test_proxy_mapping():
    """Test proxy mapping from traits."""
    config = Config()
    rng = np.random.default_rng(42)
    
    # Create sample traits
    traits = pd.DataFrame({
        'gamma': [2.0, 1.5, 2.5],
        'beta': [0.9, 0.8, 0.95],
        'kappa': [0.5, 0.3, 0.7],
        'omega': [0.0, -0.2, 0.3],
        'eta': [0.7, 0.6, 0.8]
    })
    
    proxies = map_traits_to_proxies(traits, config.proxies, rng)
    
    assert len(proxies) == 3
    expected_cols = {'night_active_ratio', 'session_std', 'task_completion_ratio', 'spending_volatility'}
    assert set(proxies.columns) == expected_cols


def test_environment_cycles():
    """Test environment cycle generation."""
    config = Config()
    rng = np.random.default_rng(42)
    
    env_series = build_sine_env(12, config.environment, rng)  # 1 year
    
    assert len(env_series.E_t) == 12
    assert len(env_series.r_t) == 12
    assert len(env_series.q_t) == 12
    assert len(env_series.macro_neg_t) == 12
    
    # Check bounds
    assert all(env_series.r_t > 0)  # Positive interest rates
    assert all(env_series.q_t > 0) and all(env_series.q_t < 1)  # Valid approval rates
    assert all(env_series.macro_neg_t >= 0)  # Non-negative macro indicators


def test_dgp_default_risk():
    """Test default risk data generation process."""
    config = Config()
    
    # Create sample data
    data = pd.DataFrame({
        'dti': [0.3, 0.5, 0.7],
        'macro_neg': [0.1, 0.2, 0.15],
        'beta': [0.9, 0.8, 0.85],
        'kappa': [0.4, 0.6, 0.5],
        'gamma': [2.0, 1.8, 2.2],
        'rate_m': [0.01, 0.012, 0.011],
        'prior_defaults': [0, 1, 0]
    })
    
    # Test PD calculation
    pds = true_pd_vectorized(data, config.dgp.logit_coefs)
    
    assert len(pds) == 3
    assert all(pds >= 0) and all(pds <= 1)  # Valid probabilities
    
    # Higher DTI should generally lead to higher PD
    assert pds[2] > pds[0]  # DTI 0.7 > DTI 0.3
    
    # Test default generation
    rng = np.random.default_rng(42)
    defaults = generate_defaults(data, config.dgp.logit_coefs, rng)
    
    assert len(defaults) == 3
    assert all(d in [0, 1] for d in defaults)  # Binary outcomes


def test_small_simulation():
    """Test running a small simulation."""
    # Create small config for fast testing
    config = Config()
    config.population.N = 50  # Small population
    config.timeline.T = 6     # Short timeline
    
    rng = np.random.default_rng(42)
    
    # Run simulation
    events_df = simulate_events(config, rng)
    
    # Basic checks
    assert len(events_df) > 0  # Should generate some events
    assert 'default' in events_df.columns
    assert 'dti' in events_df.columns
    assert events_df['id'].nunique() <= config.population.N
    assert events_df['t'].max() <= config.timeline.T
    
    # Check default rate is reasonable
    default_rate = events_df['default'].mean()
    assert 0.01 <= default_rate <= 0.5  # Reasonable range


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
