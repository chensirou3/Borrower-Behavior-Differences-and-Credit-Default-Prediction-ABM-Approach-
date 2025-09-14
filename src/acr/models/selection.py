"""Train/test splitting strategies."""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from acr.config.schema import Config


def train_test_split_temporal(
    X_baseline: pd.DataFrame,
    X_augmented: pd.DataFrame,
    y: pd.Series,
    events: pd.DataFrame,
    config: Config
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train/test sets using temporal or holdout strategy.
    
    Args:
        X_baseline: Baseline features
        X_augmented: Augmented features  
        y: Target variable
        events: Original events DataFrame (for temporal info)
        config: Configuration object
        
    Returns:
        Tuple of (X_train_base, X_test_base, X_train_aug, X_test_aug, y_train, y_test)
    """
    split_config = config.modeling.split
    
    if split_config.mode == "holdout":
        return _holdout_split(X_baseline, X_augmented, y, split_config.test_size)
    elif split_config.mode == "oot":
        return _temporal_split(X_baseline, X_augmented, y, events, split_config.test_size)
    else:
        raise ValueError(f"Unknown split mode: {split_config.mode}")


def _holdout_split(
    X_baseline: pd.DataFrame,
    X_augmented: pd.DataFrame,
    y: pd.Series,
    test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Random holdout split.
    
    Args:
        X_baseline: Baseline features
        X_augmented: Augmented features
        y: Target variable
        test_size: Fraction for test set
        
    Returns:
        Train/test split tuple
    """
    # Use sklearn's train_test_split
    indices = np.arange(len(X_baseline))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=42,
        stratify=y
    )
    
    # Split all datasets
    X_train_baseline = X_baseline.iloc[train_idx]
    X_test_baseline = X_baseline.iloc[test_idx]
    X_train_augmented = X_augmented.iloc[train_idx]
    X_test_augmented = X_augmented.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    
    return (X_train_baseline, X_test_baseline, 
            X_train_augmented, X_test_augmented, 
            y_train, y_test)


def _temporal_split(
    X_baseline: pd.DataFrame,
    X_augmented: pd.DataFrame,
    y: pd.Series,
    events: pd.DataFrame,
    test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Out-of-time (OOT) temporal split.
    
    Args:
        X_baseline: Baseline features
        X_augmented: Augmented features
        y: Target variable
        events: Events DataFrame with time information
        test_size: Fraction for test set
        
    Returns:
        Train/test split tuple
    """
    if 't' not in events.columns:
        print("Warning: No time column 't' found, falling back to holdout split")
        return _holdout_split(X_baseline, X_augmented, y, test_size)
    
    # Find temporal split point
    time_periods = sorted(events['t'].unique())
    n_periods = len(time_periods)
    split_period_idx = int(n_periods * (1 - test_size))
    split_period = time_periods[split_period_idx]
    
    # Create train/test masks
    train_mask = events['t'] < split_period
    test_mask = events['t'] >= split_period
    
    # Check that we have both train and test data
    if not train_mask.any():
        print("Warning: No training data in temporal split, falling back to holdout")
        return _holdout_split(X_baseline, X_augmented, y, test_size)
    
    if not test_mask.any():
        print("Warning: No test data in temporal split, falling back to holdout")
        return _holdout_split(X_baseline, X_augmented, y, test_size)
    
    # Split datasets
    X_train_baseline = X_baseline[train_mask]
    X_test_baseline = X_baseline[test_mask]
    X_train_augmented = X_augmented[train_mask]
    X_test_augmented = X_augmented[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"Temporal split: train periods 1-{split_period-1}, "
          f"test periods {split_period}-{time_periods[-1]}")
    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
    
    return (X_train_baseline, X_test_baseline,
            X_train_augmented, X_test_augmented,
            y_train, y_test)


def validate_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    min_samples: int = 100
) -> bool:
    """Validate train/test split.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        min_samples: Minimum samples required
        
    Returns:
        True if split is valid
    """
    # Check minimum sample sizes
    if len(X_train) < min_samples:
        print(f"Training set too small: {len(X_train)} < {min_samples}")
        return False
    
    if len(X_test) < min_samples:
        print(f"Test set too small: {len(X_test)} < {min_samples}")
        return False
    
    # Check for class imbalance
    train_pos_rate = y_train.mean()
    test_pos_rate = y_test.mean()
    
    if train_pos_rate < 0.01 or train_pos_rate > 0.99:
        print(f"Extreme class imbalance in training set: {train_pos_rate:.3f}")
        return False
    
    if test_pos_rate < 0.01 or test_pos_rate > 0.99:
        print(f"Extreme class imbalance in test set: {test_pos_rate:.3f}")
        return False
    
    # Check for feature consistency
    if not X_train.columns.equals(X_test.columns):
        print("Feature columns don't match between train and test sets")
        return False
    
    return True
