"""Classification metrics for model evaluation."""

from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_binary: np.ndarray = None
) -> Dict[str, float]:
    """Compute comprehensive classification metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        y_pred_binary: Predicted binary labels (optional)
        
    Returns:
        Dictionary of metrics
    """
    if len(y_true) == 0:
        return {}
    
    metrics = {}
    
    # AUC-ROC
    if len(np.unique(y_true)) > 1:  # Need both classes
        metrics['auc'] = float(roc_auc_score(y_true, y_pred))
        
        # PR-AUC
        metrics['pr_auc'] = float(average_precision_score(y_true, y_pred))
    else:
        metrics['auc'] = np.nan
        metrics['pr_auc'] = np.nan
    
    # Brier score (calibration)
    metrics['brier'] = float(brier_score_loss(y_true, y_pred))
    
    # KS statistic
    metrics['ks'] = compute_ks_statistic(y_true, y_pred)
    
    # Basic stats
    metrics['default_rate'] = float(np.mean(y_true))
    metrics['mean_pred'] = float(np.mean(y_pred))
    metrics['n_samples'] = int(len(y_true))
    
    return metrics


def compute_ks_statistic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Kolmogorov-Smirnov statistic.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        
    Returns:
        KS statistic
    """
    if len(np.unique(y_true)) <= 1:
        return 0.0
    
    # Get ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    
    # KS is max difference between TPR and FPR
    ks = np.max(tpr - fpr)
    
    return float(ks)


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """Compute calibration metrics and reliability curve data.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        n_bins: Number of bins for reliability curve
        
    Returns:
        Dictionary with calibration metrics and curve data
    """
    if len(y_true) == 0:
        return {}
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Compute calibration for each bin
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred[in_bin].mean()
            count_in_bin = in_bin.sum()
            
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(count_in_bin)
    
    # Expected Calibration Error (ECE)
    ece = 0.0
    for i in range(len(bin_centers)):
        ece += (bin_counts[i] / len(y_true)) * abs(bin_accuracies[i] - bin_confidences[i])
    
    return {
        'ece': float(ece),
        'bin_centers': bin_centers,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts
    }


def compute_lift_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_deciles: int = 10
) -> Dict[str, Any]:
    """Compute lift and gain metrics by deciles.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        n_deciles: Number of deciles
        
    Returns:
        Dictionary with lift metrics
    """
    if len(y_true) == 0:
        return {}
    
    # Sort by prediction score (descending)
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    
    # Split into deciles
    decile_size = len(y_true) // n_deciles
    
    deciles = []
    cumulative_gain = []
    cumulative_lift = []
    
    base_rate = np.mean(y_true)
    
    for i in range(n_deciles):
        start_idx = i * decile_size
        end_idx = (i + 1) * decile_size if i < n_deciles - 1 else len(y_true)
        
        decile_true = y_true_sorted[start_idx:end_idx]
        decile_pred = y_pred_sorted[start_idx:end_idx]
        
        decile_rate = np.mean(decile_true)
        decile_lift = decile_rate / base_rate if base_rate > 0 else 0
        
        # Cumulative metrics
        cum_true = y_true_sorted[:end_idx]
        cum_rate = np.mean(cum_true)
        cum_lift = cum_rate / base_rate if base_rate > 0 else 0
        cum_gain = np.sum(cum_true) / np.sum(y_true) if np.sum(y_true) > 0 else 0
        
        deciles.append({
            'decile': i + 1,
            'size': len(decile_true),
            'positives': int(np.sum(decile_true)),
            'rate': float(decile_rate),
            'lift': float(decile_lift),
            'avg_score': float(np.mean(decile_pred))
        })
        
        cumulative_gain.append(float(cum_gain))
        cumulative_lift.append(float(cum_lift))
    
    return {
        'deciles': deciles,
        'cumulative_gain': cumulative_gain,
        'cumulative_lift': cumulative_lift,
        'base_rate': float(base_rate)
    }
