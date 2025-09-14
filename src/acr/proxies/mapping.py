"""Trait to proxy mapping with weak correlations and noise."""

from typing import Dict

import numpy as np
import pandas as pd

from acr.config.schema import ProxiesConfig, ProxyMappingConfig


def map_traits_to_proxies(
    traits: pd.DataFrame, 
    config: ProxiesConfig, 
    rng: np.random.Generator
) -> pd.DataFrame:
    """Map traits to behavioral proxy variables with weak correlations and noise.
    
    Each proxy is computed as a linear combination of traits plus Gaussian noise:
    proxy = intercept + sum(coef_i * trait_i) + noise
    
    Args:
        traits: DataFrame with trait columns (gamma, beta, kappa, omega, eta)
        config: Proxy configuration
        rng: Random number generator
        
    Returns:
        DataFrame with proxy columns
    """
    N = len(traits)
    proxies = {}
    
    # Generate each proxy
    for proxy_name, mapping_config in config.mapping.items():
        proxy_values = _compute_single_proxy(traits, mapping_config, config.noise_sd, rng)
        proxies[proxy_name] = proxy_values
    
    return pd.DataFrame(proxies)


def _compute_single_proxy(
    traits: pd.DataFrame,
    mapping_config: ProxyMappingConfig,
    noise_sd: float,
    rng: np.random.Generator
) -> np.ndarray:
    """Compute a single proxy variable.
    
    Args:
        traits: DataFrame with trait columns
        mapping_config: Configuration for this proxy mapping
        noise_sd: Standard deviation of noise
        rng: Random number generator
        
    Returns:
        Array of proxy values
    """
    N = len(traits)
    
    # Start with intercept
    values = np.full(N, mapping_config.intercept)
    
    # Add trait contributions
    trait_coefficients = {
        'gamma': mapping_config.gamma,
        'beta': mapping_config.beta,
        'kappa': mapping_config.kappa,
        'omega': mapping_config.omega,
        'eta': mapping_config.eta
    }
    
    for trait_name, coefficient in trait_coefficients.items():
        if coefficient != 0.0 and trait_name in traits.columns:
            values += coefficient * traits[trait_name].values
    
    # Add Gaussian noise
    noise = rng.normal(0, noise_sd, N)
    values += noise
    
    # Apply constraints
    if mapping_config.clip is not None:
        min_val, max_val = mapping_config.clip
        values = np.clip(values, min_val, max_val)
    
    if mapping_config.min is not None:
        values = np.maximum(values, mapping_config.min)
    
    return values


def get_proxy_trait_correlations(
    traits: pd.DataFrame, 
    proxies: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """Compute correlations between proxies and traits.
    
    Args:
        traits: DataFrame with trait columns
        proxies: DataFrame with proxy columns
        
    Returns:
        Nested dictionary: proxy_name -> trait_name -> correlation
    """
    correlations = {}
    
    for proxy_name in proxies.columns:
        correlations[proxy_name] = {}
        for trait_name in traits.columns:
            corr = np.corrcoef(proxies[proxy_name], traits[trait_name])[0, 1]
            correlations[proxy_name][trait_name] = corr
    
    return correlations


def validate_proxy_correlations(
    correlations: Dict[str, Dict[str, float]],
    config: ProxiesConfig,
    tolerance: float = 0.3
) -> Dict[str, bool]:
    """Validate that proxy correlations are within expected ranges.
    
    Args:
        correlations: Correlation dictionary from get_proxy_trait_correlations
        config: Proxy configuration
        tolerance: Tolerance for correlation validation
        
    Returns:
        Dictionary mapping proxy names to validation status
    """
    validation_results = {}
    
    for proxy_name, mapping_config in config.mapping.items():
        if proxy_name not in correlations:
            validation_results[proxy_name] = False
            continue
            
        # Check if any strong correlations exist where expected
        expected_strong_corr = False
        actual_strong_corr = False
        
        trait_coefficients = {
            'gamma': mapping_config.gamma,
            'beta': mapping_config.beta,
            'kappa': mapping_config.kappa,
            'omega': mapping_config.omega,
            'eta': mapping_config.eta
        }
        
        for trait_name, coefficient in trait_coefficients.items():
            if abs(coefficient) > 0.1:  # Non-trivial coefficient
                expected_strong_corr = True
                if trait_name in correlations[proxy_name]:
                    actual_corr = abs(correlations[proxy_name][trait_name])
                    if actual_corr > tolerance:
                        actual_strong_corr = True
                        break
        
        # Proxy is valid if it has expected correlations or no strong expectations
        validation_results[proxy_name] = not expected_strong_corr or actual_strong_corr
    
    return validation_results
