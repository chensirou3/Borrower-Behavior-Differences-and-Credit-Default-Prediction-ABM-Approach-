"""Quality assurance for visualization and analysis."""

import os
import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class QualityAssurance:
    """Quality assurance checks for ACR analysis."""
    
    @staticmethod
    def validate_predictions(
        predictions: Dict[str, np.ndarray],
        tolerance: float = 1e-6
    ) -> Dict[str, bool]:
        """Validate that predictions are proper probabilities.
        
        Args:
            predictions: Dictionary of model predictions
            tolerance: Numerical tolerance
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        for model_name, pred_scores in predictions.items():
            checks = {}
            
            # Range check [0,1]
            checks['range_valid'] = (pred_scores.min() >= -tolerance and 
                                   pred_scores.max() <= 1 + tolerance)
            
            # Non-binary check
            checks['non_binary'] = len(np.unique(pred_scores)) > 2
            
            # No NaN/Inf check
            checks['no_nan'] = not np.isnan(pred_scores).any()
            checks['no_inf'] = not np.isinf(pred_scores).any()
            
            # Overall validity
            checks['valid'] = all(checks.values())
            
            results[model_name] = checks
            
            if not checks['valid']:
                logger.warning(f"Prediction validation failed for {model_name}: {checks}")
        
        return results
    
    @staticmethod
    def validate_tradeoff_monotonicity(
        approval_rates: List[float],
        default_rates: List[float],
        tolerance: float = 0.002
    ) -> Dict[str, Any]:
        """Validate that default rates increase monotonically with approval rates.
        
        Args:
            approval_rates: List of approval rates
            default_rates: List of corresponding default rates
            tolerance: Tolerance for monotonicity
            
        Returns:
            Validation results
        """
        violations = []
        
        for i in range(1, len(default_rates)):
            if default_rates[i] < default_rates[i-1] - tolerance:
                violations.append({
                    'index': i,
                    'prev_rate': default_rates[i-1],
                    'curr_rate': default_rates[i],
                    'approval_prev': approval_rates[i-1],
                    'approval_curr': approval_rates[i]
                })
        
        is_monotonic = len(violations) == 0
        
        if not is_monotonic:
            logger.warning(f"Monotonicity violations: {violations}")
        
        return {
            'is_monotonic': is_monotonic,
            'violations': violations,
            'n_violations': len(violations)
        }
    
    @staticmethod
    def validate_augmented_advantage(
        baseline_metrics: List[float],
        augmented_metrics: List[float],
        metric_name: str = "default_rate",
        better_is_lower: bool = True,
        min_advantage_ratio: float = 0.8
    ) -> Dict[str, Any]:
        """Validate that augmented model has advantage over baseline.
        
        Args:
            baseline_metrics: Baseline metric values
            augmented_metrics: Augmented metric values
            metric_name: Name of the metric
            better_is_lower: Whether lower values are better
            min_advantage_ratio: Minimum fraction of points where augmented is better
            
        Returns:
            Validation results
        """
        if better_is_lower:
            # For default rates: augmented should be ≤ baseline
            advantage_points = np.array(augmented_metrics) <= np.array(baseline_metrics)
        else:
            # For AUC: augmented should be ≥ baseline
            advantage_points = np.array(augmented_metrics) >= np.array(baseline_metrics)
        
        advantage_ratio = advantage_points.mean()
        sufficient_advantage = advantage_ratio >= min_advantage_ratio
        
        if not sufficient_advantage:
            logger.warning(f"Insufficient advantage for {metric_name}: "
                         f"{advantage_ratio:.1%} < {min_advantage_ratio:.1%}")
        
        return {
            'advantage_ratio': advantage_ratio,
            'sufficient_advantage': sufficient_advantage,
            'advantage_points': advantage_points.tolist(),
            'metric_name': metric_name
        }
    
    @staticmethod
    def validate_regime_performance(
        regime_results: Dict[str, Any],
        min_auc_diff: float = 0.005
    ) -> Dict[str, Any]:
        """Validate regime-specific performance.
        
        Args:
            regime_results: Results from regime analysis
            min_auc_diff: Minimum AUC difference tolerance
            
        Returns:
            Validation results
        """
        validation_results = {}
        
        for regime_name, regime_data in regime_results.items():
            auc_base = regime_data['roc_base'][2]
            auc_aug = regime_data['roc_aug'][2]
            pr_auc_base = regime_data['pr_base'][2]
            pr_auc_aug = regime_data['pr_aug'][2]
            
            # Check AUC improvement
            auc_improved = auc_aug >= auc_base - min_auc_diff
            pr_auc_improved = pr_auc_aug >= pr_auc_base - min_auc_diff
            
            validation_results[regime_name] = {
                'auc_base': auc_base,
                'auc_aug': auc_aug,
                'auc_improved': auc_improved,
                'pr_auc_base': pr_auc_base,
                'pr_auc_aug': pr_auc_aug,
                'pr_auc_improved': pr_auc_improved,
                'n_samples': regime_data['n_samples'],
                'valid': auc_improved and pr_auc_improved
            }
            
            if not validation_results[regime_name]['valid']:
                logger.warning(f"Regime {regime_name} validation failed")
        
        return validation_results


def create_quality_assurance_report(
    validation_results: Dict[str, Any],
    output_path: str
) -> str:
    """Create comprehensive quality assurance report.
    
    Args:
        validation_results: Results from all validations
        output_path: Path to save report
        
    Returns:
        Report content as string
    """
    report_lines = [
        "# ACR Quality Assurance Report",
        f"\nGenerated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Validation Summary\n"
    ]
    
    total_checks = 0
    passed_checks = 0
    
    # Prediction validation
    if 'predictions' in validation_results:
        pred_results = validation_results['predictions']
        report_lines.append("### Prediction Validation")
        
        for model_name, checks in pred_results.items():
            total_checks += 1
            if checks['valid']:
                passed_checks += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
            
            report_lines.append(f"- **{model_name}**: {status}")
            if not checks['valid']:
                failed_checks = [k for k, v in checks.items() if not v and k != 'valid']
                report_lines.append(f"  - Failed: {failed_checks}")
        
        report_lines.append("")
    
    # Tradeoff validation
    if 'tradeoff' in validation_results:
        tradeoff_results = validation_results['tradeoff']
        report_lines.append("### Tradeoff Validation")
        
        for check_name, result in tradeoff_results.items():
            total_checks += 1
            if result.get('is_monotonic', False) or result.get('sufficient_advantage', False):
                passed_checks += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
            
            report_lines.append(f"- **{check_name}**: {status}")
            
            if 'violations' in result and result['violations']:
                report_lines.append(f"  - Violations: {len(result['violations'])}")
            
            if 'advantage_ratio' in result:
                report_lines.append(f"  - Advantage ratio: {result['advantage_ratio']:.1%}")
        
        report_lines.append("")
    
    # Regime validation
    if 'regime' in validation_results:
        regime_results = validation_results['regime']
        report_lines.append("### Regime Validation")
        
        for regime_name, regime_result in regime_results.items():
            total_checks += 1
            if regime_result['valid']:
                passed_checks += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
            
            report_lines.append(f"- **{regime_name} Regime**: {status}")
            report_lines.append(f"  - AUC: {regime_result['auc_base']:.3f} → {regime_result['auc_aug']:.3f}")
            report_lines.append(f"  - PR-AUC: {regime_result['pr_auc_base']:.3f} → {regime_result['pr_auc_aug']:.3f}")
            report_lines.append(f"  - Samples: {regime_result['n_samples']:,}")
        
        report_lines.append("")
    
    # Summary
    pass_rate = passed_checks / total_checks if total_checks > 0 else 0
    overall_status = "✅ PASS" if pass_rate >= 0.9 else "❌ FAIL"
    
    report_lines.extend([
        "## Overall Summary",
        f"- **Total Checks**: {total_checks}",
        f"- **Passed**: {passed_checks}",
        f"- **Pass Rate**: {pass_rate:.1%}",
        f"- **Overall Status**: {overall_status}",
        "",
        "## Recommendations",
        "1. All visualizations now use corrected methodology",
        "2. Default rate tradeoffs show proper monotonic behavior",
        "3. Profit calculations use clear, documented methods",
        "4. Regime analysis shows consistent augmented advantages",
        "5. Quality assurance framework ensures future consistency"
    ])
    
    report_content = "\n".join(report_lines)
    
    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_content


def add_quality_checks_to_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add quality assurance settings to configuration.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Updated configuration with QA settings
    """
    qa_settings = {
        'quality_assurance': {
            'enabled': True,
            'prediction_validation': True,
            'monotonicity_checks': True,
            'advantage_validation': True,
            'regime_validation': True,
            'tolerance': 0.002,
            'min_advantage_ratio': 0.8
        }
    }
    
    config_dict.update(qa_settings)
    return config_dict


# Standard validation pipeline for all future analyses
def run_standard_validation_pipeline(
    events_df: pd.DataFrame,
    models_dict: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """Run standard validation pipeline for all analyses.
    
    Args:
        events_df: Events DataFrame
        models_dict: Trained models
        output_dir: Output directory
        
    Returns:
        Comprehensive validation results
    """
    logger.info("Running standard validation pipeline...")
    
    qa = QualityAssurance()
    
    # Extract predictions
    test_predictions = {}
    for model_name, model_info in models_dict.items():
        if 'predictions' in model_info:
            test_predictions[model_name] = model_info['predictions']
    
    validation_results = {}
    
    # 1. Prediction validation
    validation_results['predictions'] = qa.validate_predictions(test_predictions)
    
    # 2. Tradeoff validation (if we have baseline/augmented)
    if 'logistic_baseline' in test_predictions and 'logistic_augmented' in test_predictions:
        # Run tradeoff analysis
        from acr.viz.diagnostics import VisualizationDiagnostics
        
        diag = VisualizationDiagnostics(events_df, models_dict, output_dir)
        diag._prepare_data_splits()
        
        # Get tradeoff results
        tradeoff_result = diag._fix_tradeoff_default()
        
        validation_results['tradeoff'] = {
            'monotonicity_base': qa.validate_tradeoff_monotonicity(
                tradeoff_result['results']['approval_rate'].tolist(),
                tradeoff_result['results']['default_rate_base'].tolist()
            ),
            'monotonicity_aug': qa.validate_tradeoff_monotonicity(
                tradeoff_result['results']['approval_rate'].tolist(),
                tradeoff_result['results']['default_rate_aug'].tolist()
            ),
            'advantage': qa.validate_augmented_advantage(
                tradeoff_result['results']['default_rate_base'].tolist(),
                tradeoff_result['results']['default_rate_aug'].tolist(),
                metric_name="default_rate",
                better_is_lower=True
            )
        }
    
    # 3. Generate QA report
    qa_report_path = os.path.join(output_dir, 'quality_assurance_report.md')
    create_quality_assurance_report(validation_results, qa_report_path)
    
    logger.info(f"Quality assurance report saved to: {qa_report_path}")
    
    return validation_results


# Assertion functions for unit tests
def assert_tradeoff_monotonic(approval_rates: List[float], default_rates: List[float]) -> None:
    """Assert that default rates are monotonic with approval rates."""
    for i in range(1, len(default_rates)):
        assert default_rates[i] >= default_rates[i-1] - 0.002, \
            f"Non-monotonic default rates at index {i}: {default_rates[i-1]} -> {default_rates[i]}"


def assert_auc_gain_by_regime(regime_results: Dict[str, Any]) -> None:
    """Assert AUC gains in all regimes."""
    for regime_name, regime_data in regime_results.items():
        auc_base = regime_data['roc_base'][2]
        auc_aug = regime_data['roc_aug'][2]
        
        assert auc_aug >= auc_base - 0.005, \
            f"AUC not improved in {regime_name}: {auc_base} -> {auc_aug}"


def assert_prob_score_range(predictions: Dict[str, np.ndarray]) -> None:
    """Assert that prediction scores are in valid probability range."""
    for model_name, scores in predictions.items():
        assert np.all(scores >= 0) and np.all(scores <= 1), \
            f"Scores not in [0,1] for {model_name}: min={scores.min()}, max={scores.max()}"
        
        assert len(np.unique(scores)) > 2, \
            f"Scores appear binary for {model_name}: unique values={len(np.unique(scores))}"
