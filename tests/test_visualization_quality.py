"""Tests for visualization quality assurance."""

import numpy as np
import pandas as pd
import pytest

from acr.viz.quality_assurance import (
    QualityAssurance, assert_tradeoff_monotonic, 
    assert_auc_gain_by_regime, assert_prob_score_range
)


def test_prediction_validation():
    """Test prediction validation."""
    qa = QualityAssurance()
    
    # Valid predictions
    valid_predictions = {
        'model1': np.array([0.1, 0.3, 0.7, 0.9]),
        'model2': np.array([0.2, 0.4, 0.6, 0.8])
    }
    
    results = qa.validate_predictions(valid_predictions)
    
    assert all(result['valid'] for result in results.values())
    
    # Invalid predictions (out of range)
    invalid_predictions = {
        'bad_model': np.array([-0.1, 0.5, 1.5, 0.8])
    }
    
    results = qa.validate_predictions(invalid_predictions)
    assert not results['bad_model']['valid']


def test_tradeoff_monotonicity():
    """Test tradeoff monotonicity validation."""
    qa = QualityAssurance()
    
    # Monotonic case (correct)
    approval_rates = [0.5, 0.6, 0.7, 0.8, 0.85]
    monotonic_defaults = [0.08, 0.09, 0.10, 0.11, 0.12]
    
    result = qa.validate_tradeoff_monotonicity(approval_rates, monotonic_defaults)
    assert result['is_monotonic']
    assert result['n_violations'] == 0
    
    # Non-monotonic case (incorrect)
    non_monotonic_defaults = [0.12, 0.10, 0.11, 0.09, 0.08]  # Decreasing
    
    result = qa.validate_tradeoff_monotonicity(approval_rates, non_monotonic_defaults)
    assert not result['is_monotonic']
    assert result['n_violations'] > 0


def test_augmented_advantage():
    """Test augmented advantage validation."""
    qa = QualityAssurance()
    
    # Case where augmented is better (lower default rates)
    baseline_defaults = [0.12, 0.13, 0.14, 0.15, 0.16]
    augmented_defaults = [0.10, 0.11, 0.12, 0.13, 0.14]  # Consistently better
    
    result = qa.validate_augmented_advantage(
        baseline_defaults, augmented_defaults, 
        metric_name="default_rate", better_is_lower=True
    )
    
    assert result['sufficient_advantage']
    assert result['advantage_ratio'] == 1.0
    
    # Case where augmented is not consistently better
    mixed_augmented = [0.11, 0.12, 0.15, 0.16, 0.17]  # Worse in later points
    
    result = qa.validate_augmented_advantage(
        baseline_defaults, mixed_augmented,
        metric_name="default_rate", better_is_lower=True
    )
    
    assert not result['sufficient_advantage']


def test_assertion_functions():
    """Test assertion functions."""
    
    # Test monotonicity assertion
    approval_rates = [0.5, 0.6, 0.7, 0.8, 0.85]
    monotonic_defaults = [0.08, 0.09, 0.10, 0.11, 0.12]
    
    # Should not raise
    assert_tradeoff_monotonic(approval_rates, monotonic_defaults)
    
    # Should raise
    non_monotonic = [0.12, 0.10, 0.11, 0.09, 0.08]
    with pytest.raises(AssertionError):
        assert_tradeoff_monotonic(approval_rates, non_monotonic)
    
    # Test probability range assertion
    valid_predictions = {
        'model1': np.array([0.1, 0.3, 0.7, 0.9, 0.5, 0.2]),
        'model2': np.array([0.2, 0.4, 0.6, 0.8, 0.1, 0.9])
    }
    
    # Should not raise
    assert_prob_score_range(valid_predictions)
    
    # Should raise for out-of-range
    invalid_predictions = {
        'bad_model': np.array([-0.1, 0.5, 1.5])
    }
    with pytest.raises(AssertionError):
        assert_prob_score_range(invalid_predictions)
    
    # Should raise for binary scores
    binary_predictions = {
        'binary_model': np.array([0, 1, 0, 1, 0, 1])
    }
    with pytest.raises(AssertionError):
        assert_prob_score_range(binary_predictions)


def test_regime_validation():
    """Test regime validation."""
    qa = QualityAssurance()
    
    # Mock regime results
    regime_results = {
        'loose': {
            'roc_base': (None, None, 0.55),   # (fpr, tpr, auc)
            'roc_aug': (None, None, 0.58),
            'pr_base': (None, None, 0.12),    # (prec, rec, pr_auc)
            'pr_aug': (None, None, 0.14),
            'n_samples': 1000
        },
        'tight': {
            'roc_base': (None, None, 0.52),
            'roc_aug': (None, None, 0.50),    # Worse performance (should fail)
            'pr_base': (None, None, 0.15),
            'pr_aug': (None, None, 0.16),
            'n_samples': 1200
        }
    }
    
    results = qa.validate_regime_performance(regime_results)
    
    assert results['loose']['valid']      # Should pass
    assert not results['tight']['valid']  # Should fail (AUC decreased)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
