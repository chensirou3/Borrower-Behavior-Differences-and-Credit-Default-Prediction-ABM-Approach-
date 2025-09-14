"""Main simulation runner and event generation."""

import hashlib
import json
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from acr.config.schema import Config
from acr.traits.sampler import IndependentTraitSampler
from acr.proxies.mapping import map_traits_to_proxies
from acr.environment.cycles import build_sine_env, get_monthly_rates
from acr.agents.borrower import create_borrowers_from_data
from acr.dgp.default_risk import generate_defaults, calibrate_intercept_to_target_rate
from acr.simulation.schema import apply_events_schema


logger = logging.getLogger(__name__)


def simulate_events(config: Config, rng: np.random.Generator) -> pd.DataFrame:
    """Run the main simulation and generate events DataFrame.
    
    This is the core simulation function that:
    1. Samples traits and generates proxies
    2. Builds environment time series  
    3. Creates borrower agents
    4. Simulates loan applications and defaults over time
    5. Returns complete events DataFrame
    
    Args:
        config: Configuration object
        rng: Random number generator
        
    Returns:
        DataFrame with all loan application events
    """
    logger.info("Starting simulation...")
    
    # Step 1: Sample traits and generate proxies
    logger.info("Sampling traits...")
    trait_sampler = IndependentTraitSampler(config.traits)
    traits_df = trait_sampler.sample(config.population.N, rng)
    
    logger.info("Generating proxies...")
    proxies_df = map_traits_to_proxies(traits_df, config.proxies, rng)
    
    # Step 2: Build environment time series
    logger.info("Building environment...")
    env_series = build_sine_env(config.timeline.T, config.environment, rng)
    monthly_rates = get_monthly_rates(env_series.r_t)
    
    # Step 3: Create borrower agents
    logger.info("Creating borrower agents...")
    borrowers = create_borrowers_from_data(traits_df, proxies_df, rng=rng)
    
    # Step 4: Simulate events over time
    logger.info("Simulating events over time...")
    events = []
    
    for t in range(1, config.timeline.T + 1):
        logger.debug(f"Simulating period {t}/{config.timeline.T}")
        
        # Environment variables for this period
        env_idx = t - 1  # Convert to 0-based indexing
        rate_m = monthly_rates[env_idx]
        macro_neg = env_series.macro_neg_t[env_idx]
        
        # Application rate adjustment based on environment
        env_factor = config.application.amp_with_env * env_series.E_t[env_idx]
        
        # Process each borrower
        for borrower in borrowers:
            # Check if borrower applies for loan
            if borrower.should_apply(config.application.base_rate, env_factor, rng):
                # Generate loan amount
                loan_amount = borrower.generate_loan_amount(
                    config.loan.base_multiple_month_income,
                    config.loan.noise_sd,
                    config.loan.min_amount,
                    config.loan.max_amount,
                    rng
                )
                
                # Compute DTI
                dti = borrower.compute_dti(loan_amount)
                
                # Create event record
                event = {
                    # Basic info
                    't': t,
                    'id': borrower.id,
                    'loan': loan_amount,
                    'income_m': borrower.income_m,
                    'dti': dti,
                    'rate_m': rate_m,
                    'macro_neg': macro_neg,
                    'prior_defaults': borrower.prior_defaults,
                    
                    # Latent traits
                    'beta': borrower.beta,
                    'kappa': borrower.kappa,
                    'gamma': borrower.gamma,
                    'omega': borrower.omega,
                    'eta': borrower.eta,
                    
                    # Proxies
                    'night_active_ratio': borrower.night_active_ratio,
                    'session_std': borrower.session_std,
                    'task_completion_ratio': borrower.task_completion_ratio,
                    'spending_volatility': borrower.spending_volatility,
                    
                    # Default outcome (will be filled next)
                    'default': 0  # Placeholder
                }
                
                events.append(event)
    
    if not events:
        logger.warning("No loan applications generated!")
        return pd.DataFrame()
    
    # Convert to DataFrame
    events_df = pd.DataFrame(events)
    logger.info(f"Generated {len(events_df)} loan application events")
    
    # Step 5: Generate default outcomes
    logger.info("Generating default outcomes...")
    
    # Optionally calibrate intercept
    coefs = config.dgp.logit_coefs
    if config.dgp.target_default_rate:
        logger.info("Calibrating DGP intercept...")
        calibrated_a0, achieved_rate = calibrate_intercept_to_target_rate(
            events_df, coefs, config.dgp.target_default_rate
        )
        logger.info(f"Calibrated a0: {calibrated_a0:.3f}, achieved rate: {achieved_rate:.3f}")
        
        # Update coefficients
        coefs.a0 = calibrated_a0
    
    # Generate defaults
    defaults = generate_defaults(events_df, coefs, rng)
    events_df['default'] = defaults
    
    # Step 6: Update borrower states based on defaults
    logger.info("Updating borrower states...")
    for _, event in events_df.iterrows():
        if event['default'] == 1:
            borrower_id = int(event['id'])
            borrowers[borrower_id].record_default()
    
    # Step 7: Apply schema and return
    logger.info("Applying schema...")
    events_df = apply_events_schema(events_df)
    
    # Log summary statistics
    default_rate = events_df['default'].mean()
    avg_loan = events_df['loan'].mean()
    logger.info(f"Simulation complete: {len(events_df)} events, "
                f"{default_rate:.1%} default rate, "
                f"${avg_loan:,.0f} avg loan")
    
    return events_df


def run_simulation_with_manifest(
    config: Config,
    output_dir: str,
    rng: np.random.Generator = None
) -> Dict[str, Any]:
    """Run simulation and save with manifest.
    
    Args:
        config: Configuration object
        output_dir: Output directory path
        rng: Random number generator (will create if None)
        
    Returns:
        Dictionary with simulation results and paths
    """
    import os
    import json
    import hashlib
    from datetime import datetime
    
    if rng is None:
        rng = np.random.default_rng(config.seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run simulation
    events_df = simulate_events(config, rng)
    
    # Save events
    events_path = os.path.join(output_dir, "events.csv")
    events_df.to_csv(events_path, index=False)
    
    # Save configuration
    config_path = os.path.join(output_dir, "config.yaml")
    from acr.config.loader import save_config
    save_config(config, config_path)
    
    # Create manifest
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'seed': config.seed,
        'config_hash': _compute_config_hash(config),
        'n_events': len(events_df),
        'n_borrowers': events_df['id'].nunique() if len(events_df) > 0 else 0,
        'n_periods': config.timeline.T,
        'default_rate': float(events_df['default'].mean()) if len(events_df) > 0 else 0.0,
        'files': {
            'events': 'events.csv',
            'config': 'config.yaml'
        }
    }
    
    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return {
        'events_df': events_df,
        'manifest': manifest,
        'output_dir': output_dir,
        'events_path': events_path,
        'config_path': config_path,
        'manifest_path': manifest_path
    }


def _compute_config_hash(config: Config) -> str:
    """Compute hash of configuration for reproducibility.
    
    Args:
        config: Configuration object
        
    Returns:
        SHA256 hash string
    """
    config_dict = config.model_dump()
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]
