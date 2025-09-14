"""Borrower agent implementation."""

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd


@dataclass
class Borrower:
    """Individual borrower agent with traits and state.
    
    Represents a single borrower with:
    - Fixed traits (gamma, beta, kappa, omega, eta)
    - Time-varying state (income, DTI, prior defaults)
    - Behavioral proxies
    """
    
    # Unique identifier
    id: int
    
    # Fixed traits (latent)
    gamma: float   # Risk appetite
    beta: float    # Financial discipline
    kappa: float   # Behavioral volatility
    omega: float   # External shock sensitivity
    eta: float     # Learning/adaptation
    
    # Behavioral proxies (derived from traits)
    night_active_ratio: float
    session_std: float
    task_completion_ratio: float
    spending_volatility: float
    
    # Time-varying state
    income_m: float = 3000.0  # Monthly income
    prior_defaults: int = 0   # Number of prior defaults
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert borrower to dictionary representation.
        
        Returns:
            Dictionary with all borrower attributes
        """
        return {
            'id': self.id,
            'gamma': self.gamma,
            'beta': self.beta,
            'kappa': self.kappa,
            'omega': self.omega,
            'eta': self.eta,
            'night_active_ratio': self.night_active_ratio,
            'session_std': self.session_std,
            'task_completion_ratio': self.task_completion_ratio,
            'spending_volatility': self.spending_volatility,
            'income_m': self.income_m,
            'prior_defaults': self.prior_defaults
        }
    
    def update_income(self, new_income: float) -> None:
        """Update monthly income.
        
        Args:
            new_income: New monthly income
        """
        self.income_m = max(new_income, 100.0)  # Minimum income floor
    
    def record_default(self) -> None:
        """Record a default event."""
        self.prior_defaults += 1
    
    def compute_dti(self, loan_amount: float, other_debt: float = 0.0) -> float:
        """Compute debt-to-income ratio.
        
        Args:
            loan_amount: Proposed loan amount
            other_debt: Other existing debt
            
        Returns:
            DTI ratio
        """
        total_debt = loan_amount + other_debt
        return total_debt / (self.income_m * 12)  # Annual DTI
    
    def should_apply(
        self, 
        base_rate: float, 
        env_factor: float, 
        rng: np.random.Generator
    ) -> bool:
        """Determine if borrower should apply for loan this period.
        
        Application probability depends on:
        - Base application rate
        - Environment factor (economic conditions)
        - Individual traits (risk appetite, past experience)
        
        Args:
            base_rate: Base application rate
            env_factor: Environmental adjustment factor
            rng: Random number generator
            
        Returns:
            True if should apply, False otherwise
        """
        # Adjust base rate by environment
        app_rate = base_rate + env_factor
        
        # Individual adjustments based on traits
        # Higher gamma (risk appetite) -> more likely to apply
        gamma_adj = 0.1 * (self.gamma - 2.0)  # Centered around mean of 2.0
        
        # Prior defaults reduce application probability
        default_penalty = 0.05 * self.prior_defaults
        
        # Final application probability
        final_rate = app_rate + gamma_adj - default_penalty
        final_rate = np.clip(final_rate, 0.01, 0.99)
        
        # Random draw
        return rng.uniform() < final_rate
    
    def generate_loan_amount(
        self,
        base_multiple: float,
        noise_sd: float,
        min_amount: float,
        max_amount: float,
        rng: np.random.Generator
    ) -> float:
        """Generate desired loan amount.
        
        Args:
            base_multiple: Base multiple of monthly income
            noise_sd: Standard deviation of noise
            min_amount: Minimum loan amount
            max_amount: Maximum loan amount
            rng: Random number generator
            
        Returns:
            Desired loan amount
        """
        # Base amount from income
        base_amount = self.income_m * base_multiple
        
        # Individual adjustment based on traits
        # Higher gamma -> larger loans
        gamma_adj = 0.2 * (self.gamma - 2.0)
        adjusted_multiple = base_multiple * (1 + gamma_adj)
        adjusted_amount = self.income_m * adjusted_multiple
        
        # Add noise
        noise = rng.normal(0, noise_sd)
        final_amount = adjusted_amount + noise
        
        # Apply bounds
        final_amount = np.clip(final_amount, min_amount, max_amount)
        
        return final_amount


def create_borrowers_from_data(
    traits_df: pd.DataFrame,
    proxies_df: pd.DataFrame,
    base_income: float = 3000.0,
    income_std: float = 1000.0,
    rng: np.random.Generator = None
) -> list[Borrower]:
    """Create borrower agents from traits and proxies data.
    
    Args:
        traits_df: DataFrame with trait columns
        proxies_df: DataFrame with proxy columns
        base_income: Base monthly income
        income_std: Standard deviation of income
        rng: Random number generator
        
    Returns:
        List of Borrower objects
    """
    if rng is None:
        rng = np.random.default_rng()
    
    N = len(traits_df)
    borrowers = []
    
    # Generate incomes
    incomes = rng.normal(base_income, income_std, N)
    incomes = np.maximum(incomes, 500.0)  # Minimum income floor
    
    for i in range(N):
        borrower = Borrower(
            id=i,
            # Traits
            gamma=traits_df.iloc[i]['gamma'],
            beta=traits_df.iloc[i]['beta'],
            kappa=traits_df.iloc[i]['kappa'],
            omega=traits_df.iloc[i]['omega'],
            eta=traits_df.iloc[i]['eta'],
            # Proxies
            night_active_ratio=proxies_df.iloc[i]['night_active_ratio'],
            session_std=proxies_df.iloc[i]['session_std'],
            task_completion_ratio=proxies_df.iloc[i]['task_completion_ratio'],
            spending_volatility=proxies_df.iloc[i]['spending_volatility'],
            # State
            income_m=incomes[i],
            prior_defaults=0
        )
        borrowers.append(borrower)
    
    return borrowers


def borrowers_to_dataframe(borrowers: list[Borrower]) -> pd.DataFrame:
    """Convert list of borrowers to DataFrame.
    
    Args:
        borrowers: List of Borrower objects
        
    Returns:
        DataFrame with borrower data
    """
    data = [borrower.to_dict() for borrower in borrowers]
    return pd.DataFrame(data)
