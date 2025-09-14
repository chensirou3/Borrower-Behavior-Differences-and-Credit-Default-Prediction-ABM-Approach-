"""Bank decision policies and pricing."""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any

import numpy as np

from acr.config.schema import BankPolicyConfig


class DecisionPolicy(Protocol):
    """Protocol for bank decision policies."""
    
    def approve(
        self, 
        scores: np.ndarray, 
        mode: str, 
        q_or_tau: float
    ) -> np.ndarray:
        """Make approval decisions.
        
        Args:
            scores: Risk scores (lower = better)
            mode: Decision mode ('cap' or 'threshold')
            q_or_tau: Approval rate (cap mode) or threshold (threshold mode)
            
        Returns:
            Binary array of approval decisions (1=approve, 0=reject)
        """
        ...


class CapDecisionPolicy:
    """Cap-based decision policy (Stage 0).
    
    Approves the q_t fraction of applicants with lowest risk scores.
    """
    
    def __init__(self, config: BankPolicyConfig):
        """Initialize policy.
        
        Args:
            config: Bank policy configuration
        """
        self.config = config
    
    def approve(
        self, 
        scores: np.ndarray, 
        mode: str, 
        q_or_tau: float
    ) -> np.ndarray:
        """Make approval decisions using cap approach.
        
        Args:
            scores: Risk scores (lower = better)
            mode: Decision mode (must be 'cap' for this policy)
            q_or_tau: Approval rate (fraction to approve)
            
        Returns:
            Binary array of approval decisions
        """
        if mode != 'cap':
            raise ValueError(f"CapDecisionPolicy only supports 'cap' mode, got '{mode}'")
        
        if len(scores) == 0:
            return np.array([])
        
        # Number to approve
        n_approve = int(np.round(len(scores) * q_or_tau))
        n_approve = max(0, min(n_approve, len(scores)))
        
        if n_approve == 0:
            return np.zeros(len(scores), dtype=int)
        
        if n_approve == len(scores):
            return np.ones(len(scores), dtype=int)
        
        # Find threshold score for top q_t fraction
        threshold_score = np.partition(scores, n_approve - 1)[n_approve - 1]
        
        # Approve all with scores <= threshold
        # Handle ties by random selection
        approvals = np.zeros(len(scores), dtype=int)
        better_mask = scores < threshold_score
        equal_mask = scores == threshold_score
        
        # Approve all strictly better
        approvals[better_mask] = 1
        n_approved_so_far = better_mask.sum()
        
        # Randomly approve some of the ties
        n_ties = equal_mask.sum()
        n_ties_to_approve = n_approve - n_approved_so_far
        
        if n_ties_to_approve > 0 and n_ties > 0:
            tie_indices = np.where(equal_mask)[0]
            # Random selection without replacement
            rng = np.random.default_rng()
            selected_ties = rng.choice(
                tie_indices, 
                size=min(n_ties_to_approve, n_ties), 
                replace=False
            )
            approvals[selected_ties] = 1
        
        return approvals


class ThresholdDecisionPolicy:
    """Threshold-based decision policy (Stage 1+).
    
    Approves all applicants with risk scores below threshold.
    """
    
    def __init__(self, config: BankPolicyConfig):
        """Initialize policy.
        
        Args:
            config: Bank policy configuration
        """
        self.config = config
        raise NotImplementedError("ThresholdDecisionPolicy will be implemented in Stage 1")
    
    def approve(
        self, 
        scores: np.ndarray, 
        mode: str, 
        q_or_tau: float
    ) -> np.ndarray:
        """Make approval decisions using threshold approach.
        
        Args:
            scores: Risk scores (lower = better)
            mode: Decision mode (must be 'threshold' for this policy)
            q_or_tau: Risk score threshold
            
        Returns:
            Binary array of approval decisions
        """
        raise NotImplementedError("Will be implemented in Stage 1")


class Pricing(Protocol):
    """Protocol for loan pricing."""
    
    def price(self, base_rate: float, pd_hat: float) -> float:
        """Compute loan price.
        
        Args:
            base_rate: Base interest rate
            pd_hat: Estimated probability of default
            
        Returns:
            Loan interest rate
        """
        ...


class SimplePricing:
    """Simple pricing without risk premium (Stage 0)."""
    
    def __init__(self, config: BankPolicyConfig):
        """Initialize pricing.
        
        Args:
            config: Bank policy configuration
        """
        self.config = config
    
    def price(self, base_rate: float, pd_hat: float) -> float:
        """Compute loan price without risk premium.
        
        Args:
            base_rate: Base interest rate
            pd_hat: Estimated probability of default (ignored in Stage 0)
            
        Returns:
            Loan interest rate (just base + spread)
        """
        return base_rate + self.config.pricing.r0_spread_annual


class RiskBasedPricing:
    """Risk-based pricing (Stage 1+)."""
    
    def __init__(self, config: BankPolicyConfig):
        """Initialize pricing.
        
        Args:
            config: Bank policy configuration
        """
        self.config = config
        raise NotImplementedError("RiskBasedPricing will be implemented in Stage 1")
    
    def price(self, base_rate: float, pd_hat: float) -> float:
        """Compute risk-adjusted loan price.
        
        Args:
            base_rate: Base interest rate
            pd_hat: Estimated probability of default
            
        Returns:
            Risk-adjusted loan interest rate
        """
        raise NotImplementedError("Will be implemented in Stage 1")


class CapitalConstraint(Protocol):
    """Protocol for capital constraints."""
    
    def update(
        self, 
        realized_defaults: np.ndarray, 
        approvals: np.ndarray, 
        losses: np.ndarray
    ) -> Dict[str, Any]:
        """Update capital constraint state.
        
        Args:
            realized_defaults: Array of realized defaults
            approvals: Array of approvals made
            losses: Array of losses incurred
            
        Returns:
            Dictionary with constraint status and adjustments
        """
        ...


class SimpleCapitalConstraint:
    """Simple capital constraint (Stage 1+)."""
    
    def __init__(self, config: BankPolicyConfig):
        """Initialize constraint.
        
        Args:
            config: Bank policy configuration
        """
        self.config = config
        raise NotImplementedError("SimpleCapitalConstraint will be implemented in Stage 1")
    
    def update(
        self, 
        realized_defaults: np.ndarray, 
        approvals: np.ndarray, 
        losses: np.ndarray
    ) -> Dict[str, Any]:
        """Update capital constraint state.
        
        Args:
            realized_defaults: Array of realized defaults
            approvals: Array of approvals made
            losses: Array of losses incurred
            
        Returns:
            Dictionary with constraint status and adjustments
        """
        raise NotImplementedError("Will be implemented in Stage 1")


def create_decision_policy(config: BankPolicyConfig) -> DecisionPolicy:
    """Factory function to create decision policy.
    
    Args:
        config: Bank policy configuration
        
    Returns:
        Decision policy instance
    """
    if config.decision_mode == "cap":
        return CapDecisionPolicy(config)
    elif config.decision_mode == "threshold":
        return ThresholdDecisionPolicy(config)
    else:
        raise ValueError(f"Unknown decision mode: {config.decision_mode}")


def create_pricing_policy(config: BankPolicyConfig) -> Pricing:
    """Factory function to create pricing policy.
    
    Args:
        config: Bank policy configuration
        
    Returns:
        Pricing policy instance
    """
    if config.pricing.phi_per_pd == 0.0:
        return SimplePricing(config)
    else:
        return RiskBasedPricing(config)


def create_capital_constraint(config: BankPolicyConfig) -> CapitalConstraint:
    """Factory function to create capital constraint.
    
    Args:
        config: Bank policy configuration
        
    Returns:
        Capital constraint instance
    """
    if not config.capital.enabled:
        return None
    else:
        return SimpleCapitalConstraint(config)
