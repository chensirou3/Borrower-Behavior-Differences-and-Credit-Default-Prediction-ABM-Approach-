"""Fairness metrics and analysis."""

from typing import Dict, Any

import numpy as np
from sklearn.metrics import confusion_matrix


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Compute fairness metrics across groups.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        groups: Group membership (0/1)
        threshold: Decision threshold
        
    Returns:
        Dictionary of fairness metrics
    """
    if len(y_true) == 0:
        return {}
    
    # Convert predictions to binary
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Split by groups
    group_0_mask = groups == 0
    group_1_mask = groups == 1
    
    if not group_0_mask.any() or not group_1_mask.any():
        return {'error': 'Need both groups for fairness analysis'}
    
    # Compute metrics for each group
    metrics_0 = _compute_group_metrics(
        y_true[group_0_mask], 
        y_pred[group_0_mask], 
        y_pred_binary[group_0_mask]
    )
    metrics_1 = _compute_group_metrics(
        y_true[group_1_mask], 
        y_pred[group_1_mask], 
        y_pred_binary[group_1_mask]
    )
    
    # Fairness gaps
    fairness_metrics = {
        'group_0': metrics_0,
        'group_1': metrics_1,
        'gaps': {}
    }
    
    # Demographic Parity (DP) gap
    dp_gap = metrics_1['positive_rate'] - metrics_0['positive_rate']
    fairness_metrics['gaps']['demographic_parity'] = float(dp_gap)
    
    # Equal Opportunity (EO) gap - TPR difference
    eo_gap = metrics_1['tpr'] - metrics_0['tpr']
    fairness_metrics['gaps']['equal_opportunity'] = float(eo_gap)
    
    # Equalized Odds gap - max of TPR and FPR differences
    fpr_gap = metrics_1['fpr'] - metrics_0['fpr']
    eq_odds_gap = max(abs(eo_gap), abs(fpr_gap))
    fairness_metrics['gaps']['equalized_odds'] = float(eq_odds_gap)
    
    # Calibration gap
    cal_gap = metrics_1['mean_pred'] - metrics_0['mean_pred']
    fairness_metrics['gaps']['calibration'] = float(cal_gap)
    
    return fairness_metrics


def _compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_binary: np.ndarray
) -> Dict[str, float]:
    """Compute metrics for a single group.
    
    Args:
        y_true: True labels for group
        y_pred: Predicted probabilities for group
        y_pred_binary: Binary predictions for group
        
    Returns:
        Dictionary of group metrics
    """
    if len(y_true) == 0:
        return {}
    
    # Basic rates
    positive_rate = float(np.mean(y_pred_binary))
    base_rate = float(np.mean(y_true))
    mean_pred = float(np.mean(y_pred))
    
    # Confusion matrix
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred_binary)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        # Rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
        
        # Overall accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    else:
        tpr = fpr = tnr = fnr = accuracy = 0.0
    
    return {
        'n_samples': int(len(y_true)),
        'positive_rate': positive_rate,
        'base_rate': base_rate,
        'mean_pred': mean_pred,
        'tpr': float(tpr),
        'fpr': float(fpr),
        'tnr': float(tnr),
        'fnr': float(fnr),
        'accuracy': float(accuracy)
    }


def assess_fairness_violations(
    fairness_metrics: Dict[str, Any],
    thresholds: Dict[str, float] = None
) -> Dict[str, bool]:
    """Assess whether fairness violations occur.
    
    Args:
        fairness_metrics: Output from compute_fairness_metrics
        thresholds: Fairness violation thresholds
        
    Returns:
        Dictionary of violation flags
    """
    if thresholds is None:
        thresholds = {
            'demographic_parity': 0.1,   # 10% difference
            'equal_opportunity': 0.1,    # 10% TPR difference
            'equalized_odds': 0.1,       # 10% max difference
            'calibration': 0.05          # 5% calibration difference
        }
    
    violations = {}
    
    if 'gaps' in fairness_metrics:
        gaps = fairness_metrics['gaps']
        
        for metric, threshold in thresholds.items():
            if metric in gaps:
                violations[metric] = abs(gaps[metric]) > threshold
            else:
                violations[metric] = False
    
    return violations


def compute_fairness_summary(fairness_metrics: Dict[str, Any]) -> str:
    """Generate a summary report of fairness metrics.
    
    Args:
        fairness_metrics: Output from compute_fairness_metrics
        
    Returns:
        Formatted fairness report
    """
    if 'error' in fairness_metrics:
        return f"Fairness analysis error: {fairness_metrics['error']}"
    
    if 'gaps' not in fairness_metrics:
        return "No fairness gaps computed"
    
    gaps = fairness_metrics['gaps']
    group_0 = fairness_metrics.get('group_0', {})
    group_1 = fairness_metrics.get('group_1', {})
    
    lines = [
        "=== Fairness Analysis Summary ===",
        f"Group 0 size: {group_0.get('n_samples', 0)}",
        f"Group 1 size: {group_1.get('n_samples', 0)}",
        "",
        "Group Metrics:",
        f"  Base rates: {group_0.get('base_rate', 0):.3f} vs {group_1.get('base_rate', 0):.3f}",
        f"  Positive rates: {group_0.get('positive_rate', 0):.3f} vs {group_1.get('positive_rate', 0):.3f}",
        f"  TPR: {group_0.get('tpr', 0):.3f} vs {group_1.get('tpr', 0):.3f}",
        f"  FPR: {group_0.get('fpr', 0):.3f} vs {group_1.get('fpr', 0):.3f}",
        "",
        "Fairness Gaps:",
        f"  Demographic Parity: {gaps.get('demographic_parity', 0):+.3f}",
        f"  Equal Opportunity: {gaps.get('equal_opportunity', 0):+.3f}",
        f"  Equalized Odds: {gaps.get('equalized_odds', 0):+.3f}",
        f"  Calibration: {gaps.get('calibration', 0):+.3f}",
    ]
    
    # Add violation assessment
    violations = assess_fairness_violations(fairness_metrics)
    violation_count = sum(violations.values())
    
    lines.extend([
        "",
        f"Violations detected: {violation_count}/{len(violations)}",
    ])
    
    for metric, violated in violations.items():
        status = "❌" if violated else "✅"
        lines.append(f"  {status} {metric.replace('_', ' ').title()}")
    
    return "\n".join(lines)
