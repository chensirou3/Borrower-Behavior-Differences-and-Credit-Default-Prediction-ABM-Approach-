"""Environment cycle modeling with sine waves and AR(1) noise."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from acr.config.schema import EnvironmentConfig


@dataclass
class EnvSeries:
    """Environment time series data."""
    
    E_t: np.ndarray           # Core looseness ↔ tightness index (-1..+1)
    r_t: np.ndarray          # Interest rate series (annual)
    q_t: np.ndarray          # Approval rate cap/threshold parameter
    macro_neg_t: np.ndarray  # Macro negative indicator series


def build_sine_env(
    T: int, 
    config: EnvironmentConfig, 
    rng: np.random.Generator
) -> EnvSeries:
    """Build sine wave environment with AR(1) micro-noise.
    
    The core environment follows:
    E_t = sin(2π * t / period) + AR(1) noise
    
    From E_t, we derive:
    - r_t: Interest rates 
    - q_t: Approval parameters
    - macro_neg_t: Macro negative indicators
    
    Args:
        T: Number of time periods
        config: Environment configuration
        rng: Random number generator
        
    Returns:
        EnvSeries with all time series
    """
    if not config.sine.enabled:
        # Flat environment if sine is disabled
        E_t = np.zeros(T)
    else:
        E_t = _build_sine_with_ar1_noise(T, config.sine, rng)
    
    # Derive other series from E_t
    r_t = _derive_interest_rates(E_t, config.interest)
    q_t = _derive_approval_rates(E_t, config.approval)
    macro_neg_t = _derive_macro_negative(E_t, config.macro_neg)
    
    return EnvSeries(
        E_t=E_t,
        r_t=r_t,
        q_t=q_t,
        macro_neg_t=macro_neg_t
    )


def _build_sine_with_ar1_noise(
    T: int,
    sine_config,
    rng: np.random.Generator
) -> np.ndarray:
    """Build sine wave with AR(1) micro-noise.
    
    Args:
        T: Number of time periods
        sine_config: Sine configuration
        rng: Random number generator
        
    Returns:
        E_t series
    """
    # Time indices (1-based to match period indexing)
    t = np.arange(1, T + 1)
    
    # Base sine wave
    sine_wave = np.sin(2 * np.pi * t / sine_config.period)
    
    # AR(1) noise process
    ar1_noise = _generate_ar1_noise(T, sine_config.ar1_rho, sine_config.noise_sd, rng)
    
    # Combine
    E_t = sine_wave + ar1_noise
    
    return E_t


def _generate_ar1_noise(
    T: int,
    rho: float,
    noise_sd: float,
    rng: np.random.Generator
) -> np.ndarray:
    """Generate AR(1) noise process.
    
    x_t = rho * x_{t-1} + epsilon_t
    where epsilon_t ~ N(0, noise_sd²)
    
    Args:
        T: Length of series
        rho: AR(1) coefficient
        noise_sd: Standard deviation of innovations
        rng: Random number generator
        
    Returns:
        AR(1) noise series
    """
    if rho == 0:
        # White noise case
        return rng.normal(0, noise_sd, T)
    
    # Generate innovations
    innovations = rng.normal(0, noise_sd, T)
    
    # Build AR(1) series
    series = np.zeros(T)
    series[0] = innovations[0]  # Initialize
    
    for t in range(1, T):
        series[t] = rho * series[t-1] + innovations[t]
    
    return series


def _derive_interest_rates(E_t: np.ndarray, interest_config) -> np.ndarray:
    """Derive interest rates from environment index.
    
    r_t = r_mid + r_amp * E_t
    
    Args:
        E_t: Environment index
        interest_config: Interest rate configuration
        
    Returns:
        Annual interest rate series
    """
    r_mid = interest_config.r_mid_annual
    r_amp = interest_config.r_amp_annual
    
    r_t = r_mid + r_amp * E_t
    
    # Ensure non-negative rates
    r_t = np.maximum(r_t, 0.001)
    
    return r_t


def _derive_approval_rates(E_t: np.ndarray, approval_config) -> np.ndarray:
    """Derive approval parameters from environment index.
    
    q_t = q_mid - q_amp * E_t  (negative sign: tight env → lower approval)
    
    Args:
        E_t: Environment index
        approval_config: Approval configuration
        
    Returns:
        Approval parameter series
    """
    q_mid = approval_config.q_mid
    q_amp = approval_config.q_amp
    
    q_t = q_mid - q_amp * E_t
    
    # Clip to valid range
    q_t = np.clip(q_t, 0.01, 0.99)
    
    return q_t


def _derive_macro_negative(E_t: np.ndarray, macro_config) -> np.ndarray:
    """Derive macro negative indicators from environment index.
    
    macro_neg_t = m0 + m1 * max(E_t, 0)  (only positive E_t contributes)
    
    Args:
        E_t: Environment index
        macro_config: Macro negative configuration
        
    Returns:
        Macro negative indicator series
    """
    m0 = macro_config.m0
    m1 = macro_config.m1
    
    # Only positive environment contributes to macro negativity
    positive_E_t = np.maximum(E_t, 0)
    macro_neg_t = m0 + m1 * positive_E_t
    
    # Ensure non-negative
    macro_neg_t = np.maximum(macro_neg_t, 0)
    
    return macro_neg_t


def get_monthly_rates(annual_rates: np.ndarray) -> np.ndarray:
    """Convert annual rates to monthly rates.
    
    Args:
        annual_rates: Annual interest rates
        
    Returns:
        Monthly interest rates
    """
    return annual_rates / 12.0


def summarize_environment(env_series: EnvSeries) -> dict:
    """Summarize environment statistics.
    
    Args:
        env_series: Environment series
        
    Returns:
        Summary statistics dictionary
    """
    return {
        'E_t': {
            'mean': float(np.mean(env_series.E_t)),
            'std': float(np.std(env_series.E_t)),
            'min': float(np.min(env_series.E_t)),
            'max': float(np.max(env_series.E_t))
        },
        'r_t': {
            'mean': float(np.mean(env_series.r_t)),
            'std': float(np.std(env_series.r_t)),
            'min': float(np.min(env_series.r_t)),
            'max': float(np.max(env_series.r_t))
        },
        'q_t': {
            'mean': float(np.mean(env_series.q_t)),
            'std': float(np.std(env_series.q_t)),
            'min': float(np.min(env_series.q_t)),
            'max': float(np.max(env_series.q_t))
        },
        'macro_neg_t': {
            'mean': float(np.mean(env_series.macro_neg_t)),
            'std': float(np.std(env_series.macro_neg_t)),
            'min': float(np.min(env_series.macro_neg_t)),
            'max': float(np.max(env_series.macro_neg_t))
        }
    }
