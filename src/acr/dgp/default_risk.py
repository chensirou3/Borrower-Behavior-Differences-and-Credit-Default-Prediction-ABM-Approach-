"""True PD model and calibration utilities."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from acr.config.schema import DGPCoefsConfig


def true_pd_row(row: pd.Series, coefs: DGPCoefsConfig) -> float:
    """Compute true probability of default for a single row.
    
    Logistic function:
    PD = 1 / (1 + exp(-z))
    where z = a0 + a1*dti + a2*macro_neg + a3*(1-beta) + a4*kappa + 
              a5*gamma + a6*rate_m + a7*prior_defaults
    
    Args:
        row: Data row with required features
        coefs: DGP coefficient configuration
        
    Returns:
        Probability of default (0-1)
    """
    # Extract features
    dti = row['dti']
    macro_neg = row['macro_neg']
    one_minus_beta = 1.0 - row['beta']  # Higher values = worse discipline
    kappa = row['kappa']
    gamma = row['gamma']
    rate_m = row['rate_m']
    prior_defaults = row['prior_defaults']
    
    # Compute logit
    z = (coefs.a0 + 
         coefs.a1_dti * dti +
         coefs.a2_macro_neg * macro_neg +
         coefs.a3_one_minus_beta * one_minus_beta +
         coefs.a4_kappa * kappa +
         coefs.a5_gamma * gamma +
         coefs.a6_rate_m * rate_m +
         coefs.a7_prior_default * prior_defaults)
    
    # Apply logistic function with numerical stability
    if z > 500:  # Prevent overflow
        return 1.0
    elif z < -500:  # Prevent underflow
        return 0.0
    else:
        return 1.0 / (1.0 + np.exp(-z))


def true_pd_vectorized(data: pd.DataFrame, coefs: DGPCoefsConfig) -> np.ndarray:
    """Compute true PD for multiple rows (vectorized).
    
    Args:
        data: DataFrame with required features
        coefs: DGP coefficient configuration
        
    Returns:
        Array of probabilities of default
    """
    # Extract features
    dti = data['dti'].values
    macro_neg = data['macro_neg'].values
    one_minus_beta = 1.0 - data['beta'].values
    kappa = data['kappa'].values
    gamma = data['gamma'].values
    rate_m = data['rate_m'].values
    prior_defaults = data['prior_defaults'].values
    
    # Compute logit
    z = (coefs.a0 + 
         coefs.a1_dti * dti +
         coefs.a2_macro_neg * macro_neg +
         coefs.a3_one_minus_beta * one_minus_beta +
         coefs.a4_kappa * kappa +
         coefs.a5_gamma * gamma +
         coefs.a6_rate_m * rate_m +
         coefs.a7_prior_default * prior_defaults)
    
    # Apply logistic function with numerical stability
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def generate_defaults(
    data: pd.DataFrame, 
    coefs: DGPCoefsConfig, 
    rng: np.random.Generator
) -> np.ndarray:
    """Generate binary default outcomes based on true PD.
    
    Args:
        data: DataFrame with required features
        coefs: DGP coefficient configuration
        rng: Random number generator
        
    Returns:
        Binary array of default outcomes (0/1)
    """
    # Compute true PDs
    true_pds = true_pd_vectorized(data, coefs)
    
    # Generate binary outcomes
    random_draws = rng.uniform(0, 1, len(data))
    defaults = (random_draws < true_pds).astype(int)
    
    return defaults


def calibrate_intercept_to_target_rate(
    events_df: pd.DataFrame, 
    coefs: DGPCoefsConfig,
    target_range: List[float]
) -> Tuple[float, float]:
    """Calibrate intercept (a0) to achieve target default rate.
    
    Args:
        events_df: Events DataFrame with features
        coefs: DGP coefficient configuration (a0 will be ignored)
        target_range: [min_rate, max_rate] target range
        
    Returns:
        Tuple of (calibrated_a0, achieved_rate)
    """
    target_min, target_max = target_range
    target_mid = (target_min + target_max) / 2.0
    
    def objective(a0_candidate: float) -> float:
        """Objective function: squared distance from target mid-point."""
        # Create temporary config with candidate a0
        temp_coefs = DGPCoefsConfig(
            a0=a0_candidate,
            a1_dti=coefs.a1_dti,
            a2_macro_neg=coefs.a2_macro_neg,
            a3_one_minus_beta=coefs.a3_one_minus_beta,
            a4_kappa=coefs.a4_kappa,
            a5_gamma=coefs.a5_gamma,
            a6_rate_m=coefs.a6_rate_m,
            a7_prior_default=coefs.a7_prior_default
        )
        
        # Compute average PD
        pds = true_pd_vectorized(events_df, temp_coefs)
        avg_pd = np.mean(pds)
        
        # Return squared distance from target
        return (avg_pd - target_mid) ** 2
    
    # Optimize a0
    result = minimize_scalar(objective, bounds=(-10, 2), method='bounded')
    
    if not result.success:
        print(f"Warning: Calibration optimization failed. Using default a0={coefs.a0}")
        calibrated_a0 = coefs.a0
    else:
        calibrated_a0 = result.x
    
    # Compute achieved rate
    final_coefs = DGPCoefsConfig(
        a0=calibrated_a0,
        a1_dti=coefs.a1_dti,
        a2_macro_neg=coefs.a2_macro_neg,
        a3_one_minus_beta=coefs.a3_one_minus_beta,
        a4_kappa=coefs.a4_kappa,
        a5_gamma=coefs.a5_gamma,
        a6_rate_m=coefs.a6_rate_m,
        a7_prior_default=coefs.a7_prior_default
    )
    
    achieved_rate = np.mean(true_pd_vectorized(events_df, final_coefs))
    
    return calibrated_a0, achieved_rate


def validate_dgp_monotonicity(
    data_sample: pd.DataFrame,
    coefs: DGPCoefsConfig,
    feature: str,
    n_points: int = 100
) -> Dict[str, float]:
    """Validate monotonicity of DGP with respect to a feature.
    
    Args:
        data_sample: Sample data for validation
        coefs: DGP coefficients
        feature: Feature to test monotonicity for
        n_points: Number of points to test
        
    Returns:
        Dictionary with monotonicity statistics
    """
    if len(data_sample) == 0:
        return {'correlation': 0.0, 'monotonic_pairs': 0.0}
    
    # Take first row as base
    base_row = data_sample.iloc[0].copy()
    
    # Create test range for the feature
    if feature in data_sample.columns:
        feature_min = data_sample[feature].min()
        feature_max = data_sample[feature].max()
    else:
        # Default ranges for common features
        if feature == 'dti':
            feature_min, feature_max = 0.0, 1.0
        elif feature == 'macro_neg':
            feature_min, feature_max = 0.0, 0.5
        else:
            feature_min, feature_max = 0.0, 1.0
    
    test_values = np.linspace(feature_min, feature_max, n_points)
    test_pds = []
    
    # Compute PD for each test value
    for value in test_values:
        base_row[feature] = value
        pd_value = true_pd_row(base_row, coefs)
        test_pds.append(pd_value)
    
    test_pds = np.array(test_pds)
    
    # Check monotonicity
    correlation = np.corrcoef(test_values, test_pds)[0, 1]
    
    # Count monotonic pairs
    monotonic_count = 0
    total_pairs = 0
    
    for i in range(len(test_values)):
        for j in range(i + 1, len(test_values)):
            total_pairs += 1
            if (test_values[j] > test_values[i] and test_pds[j] >= test_pds[i]):
                monotonic_count += 1
    
    monotonic_fraction = monotonic_count / total_pairs if total_pairs > 0 else 0.0
    
    return {
        'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
        'monotonic_pairs': float(monotonic_fraction)
    }


def compute_dgp_summary_stats(
    events_df: pd.DataFrame,
    coefs: DGPCoefsConfig
) -> Dict[str, float]:
    """Compute summary statistics for the DGP.
    
    Args:
        events_df: Events DataFrame
        coefs: DGP coefficients
        
    Returns:
        Dictionary of summary statistics
    """
    if len(events_df) == 0:
        return {}
    
    # Compute PDs
    pds = true_pd_vectorized(events_df, coefs)
    
    # Basic statistics
    stats = {
        'mean_pd': float(np.mean(pds)),
        'std_pd': float(np.std(pds)),
        'min_pd': float(np.min(pds)),
        'max_pd': float(np.max(pds)),
        'median_pd': float(np.median(pds))
    }
    
    # Percentiles
    percentiles = [5, 25, 75, 95]
    for p in percentiles:
        stats[f'p{p}_pd'] = float(np.percentile(pds, p))
    
    # Feature-specific statistics if available
    if 'default' in events_df.columns:
        actual_defaults = events_df['default'].values
        stats['actual_default_rate'] = float(np.mean(actual_defaults))
        
        # Calibration: correlation between PD and actual defaults
        if len(np.unique(actual_defaults)) > 1:  # Need variation
            stats['pd_default_correlation'] = float(np.corrcoef(pds, actual_defaults)[0, 1])
    
    return stats
