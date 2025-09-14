"""Feature set construction for modeling."""

from typing import Tuple, Optional

import numpy as np
import pandas as pd

from acr.config.schema import Config
from acr.simulation.schema import BASELINE_FEATURES, PROXY_FEATURES


def build_datasets(
    events: pd.DataFrame, 
    config: Config
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Build baseline and augmented feature datasets.
    
    Args:
        events: Events DataFrame from simulation
        config: Configuration object
        
    Returns:
        Tuple of (X_baseline, X_augmented, y, group)
        - X_baseline: Baseline features
        - X_augmented: Baseline + proxy features  
        - y: Target variable (default)
        - group: Grouping variable for fairness analysis
    """
    if len(events) == 0:
        # Return empty datasets
        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype='int64')
        return empty_df, empty_df, empty_series, empty_series
    
    # Extract baseline features
    baseline_cols = [col for col in config.features.baseline if col in events.columns]
    X_baseline = events[baseline_cols].copy()
    
    # Extract proxy features
    proxy_cols = [col for col in config.features.proxies if col in events.columns]
    X_augmented = pd.concat([X_baseline, events[proxy_cols]], axis=1)
    
    # Target variable
    y = events['default'].copy()
    
    # Grouping variable for fairness analysis
    group = create_fairness_groups(events, config.evaluation.fairness.group_by)
    
    return X_baseline, X_augmented, y, group


def create_fairness_groups(events: pd.DataFrame, group_by: str) -> pd.Series:
    """Create binary groups for fairness analysis.
    
    Args:
        events: Events DataFrame
        group_by: Grouping strategy
        
    Returns:
        Binary group indicator (0/1)
    """
    if group_by == "night_active_high":
        # Group by high vs low night activity
        if 'night_active_ratio' in events.columns:
            median_val = events['night_active_ratio'].median()
            return (events['night_active_ratio'] > median_val).astype(int)
        else:
            # Fallback: random groups
            return np.random.binomial(1, 0.5, len(events))
    
    elif group_by == "spending_volatility_high":
        # Group by high vs low spending volatility
        if 'spending_volatility' in events.columns:
            median_val = events['spending_volatility'].median()
            return (events['spending_volatility'] > median_val).astype(int)
        else:
            return np.random.binomial(1, 0.5, len(events))
    
    elif group_by == "income_low":
        # Group by low vs high income
        median_income = events['income_m'].median()
        return (events['income_m'] <= median_income).astype(int)
    
    else:
        # Default: random groups
        return np.random.binomial(1, 0.5, len(events))


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features (Stage 1+).
    
    Args:
        X: Input features DataFrame
        
    Returns:
        DataFrame with additional engineered features
    """
    X_eng = X.copy()
    
    # DTI bins
    if 'dti' in X_eng.columns:
        X_eng['dti_high'] = (X_eng['dti'] > 0.4).astype(int)
        X_eng['dti_very_high'] = (X_eng['dti'] > 0.6).astype(int)
    
    # Income bins
    if 'income_m' in X_eng.columns:
        X_eng['income_low'] = (X_eng['income_m'] < 2500).astype(int)
        X_eng['income_high'] = (X_eng['income_m'] > 5000).astype(int)
    
    # Loan size relative to income
    if 'loan' in X_eng.columns and 'income_m' in X_eng.columns:
        X_eng['loan_to_income'] = X_eng['loan'] / (X_eng['income_m'] * 12)
    
    # Prior defaults indicator
    if 'prior_defaults' in X_eng.columns:
        X_eng['has_prior_defaults'] = (X_eng['prior_defaults'] > 0).astype(int)
    
    return X_eng


def get_feature_importance_names(feature_set: str, config: Config) -> list[str]:
    """Get feature names for importance analysis.
    
    Args:
        feature_set: 'baseline' or 'augmented'
        config: Configuration object
        
    Returns:
        List of feature names
    """
    if feature_set == 'baseline':
        return config.features.baseline.copy()
    elif feature_set == 'augmented':
        return config.features.baseline + config.features.proxies
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")


def validate_features(X: pd.DataFrame, feature_names: list[str]) -> bool:
    """Validate feature DataFrame.
    
    Args:
        X: Features DataFrame
        feature_names: Expected feature names
        
    Returns:
        True if valid, False otherwise
    """
    # Check for missing columns
    missing_cols = set(feature_names) - set(X.columns)
    if missing_cols:
        print(f"Missing feature columns: {missing_cols}")
        return False
    
    # Check for missing values
    missing_values = X[feature_names].isnull().sum()
    if missing_values.any():
        print(f"Features with missing values: {missing_values[missing_values > 0]}")
        return False
    
    # Check for infinite values
    for col in feature_names:
        if np.isinf(X[col]).any():
            print(f"Feature {col} contains infinite values")
            return False
    
    return True
