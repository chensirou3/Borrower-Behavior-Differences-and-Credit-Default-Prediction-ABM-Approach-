"""Diagnostics for proxy-trait relationships."""

from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from acr.config.schema import ProxiesConfig


def compute_proxy_diagnostics(
    traits: pd.DataFrame,
    proxies: pd.DataFrame,
    config: ProxiesConfig
) -> Dict[str, Any]:
    """Compute diagnostic statistics for proxy-trait relationships.
    
    Args:
        traits: DataFrame with trait columns
        proxies: DataFrame with proxy columns  
        config: Proxy configuration
        
    Returns:
        Dictionary with diagnostic results
    """
    diagnostics = {
        'correlations': {},
        'r_squared': {},
        'expected_vs_actual': {},
        'summary_stats': {}
    }
    
    # Compute correlations
    for proxy_name in proxies.columns:
        diagnostics['correlations'][proxy_name] = {}
        for trait_name in traits.columns:
            corr = np.corrcoef(proxies[proxy_name], traits[trait_name])[0, 1]
            diagnostics['correlations'][proxy_name][trait_name] = float(corr)
    
    # Compute R² for each proxy
    for proxy_name in proxies.columns:
        y = proxies[proxy_name].values
        X = traits.values
        
        # Fit linear regression
        reg = LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        
        r2 = r2_score(y, y_pred)
        diagnostics['r_squared'][proxy_name] = float(r2)
    
    # Compare expected vs actual relationships
    for proxy_name, mapping_config in config.mapping.items():
        if proxy_name not in proxies.columns:
            continue
            
        expected_coeffs = {
            'gamma': mapping_config.gamma,
            'beta': mapping_config.beta,
            'kappa': mapping_config.kappa,
            'omega': mapping_config.omega,
            'eta': mapping_config.eta
        }
        
        # Fit regression to get actual coefficients
        y = proxies[proxy_name].values
        X = traits[['gamma', 'beta', 'kappa', 'omega', 'eta']].values
        
        reg = LinearRegression()
        reg.fit(X, y)
        
        actual_coeffs = {
            trait: float(coeff) 
            for trait, coeff in zip(['gamma', 'beta', 'kappa', 'omega', 'eta'], reg.coef_)
        }
        
        diagnostics['expected_vs_actual'][proxy_name] = {
            'expected': expected_coeffs,
            'actual': actual_coeffs,
            'intercept_expected': float(mapping_config.intercept),
            'intercept_actual': float(reg.intercept_)
        }
    
    # Summary statistics
    all_correlations = []
    for proxy_name in diagnostics['correlations']:
        for trait_name in diagnostics['correlations'][proxy_name]:
            corr = diagnostics['correlations'][proxy_name][trait_name]
            if not np.isnan(corr):
                all_correlations.append(abs(corr))
    
    diagnostics['summary_stats'] = {
        'mean_abs_correlation': float(np.mean(all_correlations)) if all_correlations else 0.0,
        'max_abs_correlation': float(np.max(all_correlations)) if all_correlations else 0.0,
        'mean_r_squared': float(np.mean(list(diagnostics['r_squared'].values()))),
        'n_proxies': len(proxies.columns),
        'n_traits': len(traits.columns)
    }
    
    return diagnostics


def format_diagnostics_report(diagnostics: Dict[str, Any]) -> str:
    """Format diagnostics into a readable report.
    
    Args:
        diagnostics: Results from compute_proxy_diagnostics
        
    Returns:
        Formatted report string
    """
    report_lines = []
    report_lines.append("=== Proxy-Trait Diagnostics Report ===\n")
    
    # Summary
    summary = diagnostics['summary_stats']
    report_lines.append(f"Summary Statistics:")
    report_lines.append(f"  Number of proxies: {summary['n_proxies']}")
    report_lines.append(f"  Number of traits: {summary['n_traits']}")
    report_lines.append(f"  Mean absolute correlation: {summary['mean_abs_correlation']:.3f}")
    report_lines.append(f"  Max absolute correlation: {summary['max_abs_correlation']:.3f}")
    report_lines.append(f"  Mean R²: {summary['mean_r_squared']:.3f}")
    report_lines.append("")
    
    # Correlations by proxy
    report_lines.append("Correlations by Proxy:")
    for proxy_name, trait_corrs in diagnostics['correlations'].items():
        report_lines.append(f"  {proxy_name}:")
        for trait_name, corr in trait_corrs.items():
            report_lines.append(f"    {trait_name}: {corr:+.3f}")
        r2 = diagnostics['r_squared'].get(proxy_name, 0.0)
        report_lines.append(f"    R²: {r2:.3f}")
        report_lines.append("")
    
    # Expected vs actual coefficients
    report_lines.append("Expected vs Actual Coefficients:")
    for proxy_name, comparison in diagnostics['expected_vs_actual'].items():
        report_lines.append(f"  {proxy_name}:")
        expected = comparison['expected']
        actual = comparison['actual']
        
        for trait in expected:
            exp_val = expected[trait]
            act_val = actual[trait]
            if exp_val != 0 or abs(act_val) > 0.1:
                report_lines.append(f"    {trait}: expected={exp_val:+.2f}, actual={act_val:+.2f}")
        
        exp_int = comparison['intercept_expected']
        act_int = comparison['intercept_actual']
        report_lines.append(f"    intercept: expected={exp_int:+.2f}, actual={act_int:+.2f}")
        report_lines.append("")
    
    return "\n".join(report_lines)


def save_diagnostics(
    diagnostics: Dict[str, Any], 
    output_path: str,
    format: str = 'json'
) -> None:
    """Save diagnostics to file.
    
    Args:
        diagnostics: Results from compute_proxy_diagnostics
        output_path: Path to save file
        format: Output format ('json' or 'csv')
    """
    if format == 'json':
        import json
        with open(output_path, 'w') as f:
            json.dump(diagnostics, f, indent=2)
    
    elif format == 'csv':
        # Convert to flat format for CSV
        rows = []
        for proxy_name in diagnostics['correlations']:
            for trait_name, corr in diagnostics['correlations'][proxy_name].items():
                r2 = diagnostics['r_squared'].get(proxy_name, np.nan)
                rows.append({
                    'proxy': proxy_name,
                    'trait': trait_name,
                    'correlation': corr,
                    'r_squared': r2
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
