"""Event-level data schema definitions."""

from typing import Dict, Any

import pandas as pd


# Event schema - column names and types for events.csv
EVENT_SCHEMA = {
    # Time and ID
    't': 'int32',                    # Period (1..T)
    'id': 'int32',                   # Borrower ID
    
    # Loan application
    'loan': 'float64',               # Approved or proposed loan amount
    'income_m': 'float64',           # Monthly income
    'dti': 'float64',                # Debt-to-income ratio for this period
    'rate_m': 'float64',             # Monthly interest rate
    'macro_neg': 'float64',          # Macro negative indicator ≥ 0
    'prior_defaults': 'int32',       # Historical default count (before this period)
    
    # Outcome
    'default': 'int8',               # Binary default outcome (0/1)
    
    # Latent traits (for analysis, not modeling)
    'beta': 'float64',
    'kappa': 'float64', 
    'gamma': 'float64',
    'omega': 'float64',
    'eta': 'float64',
    
    # Behavioral proxies
    'night_active_ratio': 'float64',
    'session_std': 'float64',
    'task_completion_ratio': 'float64',
    'spending_volatility': 'float64'
}

# Baseline features (for modeling)
BASELINE_FEATURES = [
    'dti', 'income_m', 'rate_m', 'macro_neg', 'prior_defaults', 'loan'
]

# Proxy features (for augmented modeling)
PROXY_FEATURES = [
    'night_active_ratio', 'session_std', 'task_completion_ratio', 'spending_volatility'
]

# All modeling features
ALL_FEATURES = BASELINE_FEATURES + PROXY_FEATURES

# Latent traits (not for modeling)
LATENT_TRAITS = ['beta', 'kappa', 'gamma', 'omega', 'eta']

# Required columns for event generation
REQUIRED_EVENT_COLUMNS = [
    't', 'id', 'loan', 'income_m', 'dti', 'rate_m', 'macro_neg', 
    'prior_defaults', 'default'
] + LATENT_TRAITS + PROXY_FEATURES


def validate_events_schema(df: pd.DataFrame) -> bool:
    """Validate that DataFrame conforms to events schema.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Check required columns
    missing_cols = set(REQUIRED_EVENT_COLUMNS) - set(df.columns)
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return False
    
    # Check data types (basic validation)
    for col, expected_dtype in EVENT_SCHEMA.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            # Allow some flexibility in numeric types
            if expected_dtype.startswith('int') and not actual_dtype.startswith('int'):
                if not actual_dtype.startswith('float'):  # float can convert to int
                    print(f"Column {col} has wrong type: expected {expected_dtype}, got {actual_dtype}")
                    return False
            elif expected_dtype.startswith('float') and not actual_dtype.startswith(('float', 'int')):
                print(f"Column {col} has wrong type: expected {expected_dtype}, got {actual_dtype}")
                return False
    
    return True


def apply_events_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Apply event schema types to DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with correct types
    """
    df_typed = df.copy()
    
    for col, dtype in EVENT_SCHEMA.items():
        if col in df_typed.columns:
            try:
                df_typed[col] = df_typed[col].astype(dtype)
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not convert column {col} to {dtype}: {e}")
    
    return df_typed


def get_feature_columns(feature_set: str) -> list[str]:
    """Get column names for a feature set.
    
    Args:
        feature_set: 'baseline', 'proxies', or 'all'
        
    Returns:
        List of feature column names
    """
    if feature_set == 'baseline':
        return BASELINE_FEATURES.copy()
    elif feature_set == 'proxies':
        return PROXY_FEATURES.copy()
    elif feature_set == 'all':
        return ALL_FEATURES.copy()
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")


def summarize_events(df: pd.DataFrame) -> Dict[str, Any]:
    """Summarize events DataFrame.
    
    Args:
        df: Events DataFrame
        
    Returns:
        Summary statistics dictionary
    """
    if len(df) == 0:
        return {'n_events': 0}
    
    summary = {
        'n_events': len(df),
        'n_borrowers': df['id'].nunique(),
        'n_periods': df['t'].nunique(),
        'period_range': [int(df['t'].min()), int(df['t'].max())],
        'default_rate': float(df['default'].mean()),
        'avg_loan_size': float(df['loan'].mean()),
        'avg_dti': float(df['dti'].mean()),
        'avg_income': float(df['income_m'].mean()),
    }
    
    # Add feature summaries
    for feature in ALL_FEATURES:
        if feature in df.columns:
            summary[f'{feature}_mean'] = float(df[feature].mean())
            summary[f'{feature}_std'] = float(df[feature].std())
    
    return summary
