"""Bank accounting and profit tracking."""

from dataclasses import dataclass, field
from typing import Dict, List, Any

import numpy as np
import pandas as pd


@dataclass
class BankAccounting:
    """Bank accounting system for tracking profits and losses."""
    
    # Cumulative metrics
    total_loans_approved: int = 0
    total_loan_volume: float = 0.0
    total_interest_income: float = 0.0
    total_default_losses: float = 0.0
    total_net_profit: float = 0.0
    
    # Period-by-period tracking
    period_data: List[Dict[str, Any]] = field(default_factory=list)
    
    def record_period(
        self,
        period: int,
        approvals: np.ndarray,
        loan_amounts: np.ndarray,
        interest_rates: np.ndarray,
        defaults: np.ndarray,
        loan_terms: int = 12  # months
    ) -> Dict[str, float]:
        """Record accounting data for a period.
        
        Args:
            period: Time period
            approvals: Binary array of approvals (1=approved, 0=rejected)
            loan_amounts: Array of loan amounts
            interest_rates: Array of interest rates (annual)
            defaults: Binary array of defaults (1=default, 0=no default)
            loan_terms: Loan term in months
            
        Returns:
            Dictionary with period metrics
        """
        # Filter to approved loans only
        approved_mask = approvals.astype(bool)
        
        if not approved_mask.any():
            # No approvals this period
            period_metrics = {
                'period': period,
                'n_approved': 0,
                'loan_volume': 0.0,
                'interest_income': 0.0,
                'default_losses': 0.0,
                'net_profit': 0.0,
                'approval_rate': 0.0,
                'default_rate': 0.0,
                'avg_loan_size': 0.0,
                'avg_interest_rate': 0.0
            }
        else:
            approved_amounts = loan_amounts[approved_mask]
            approved_rates = interest_rates[approved_mask]
            approved_defaults = defaults[approved_mask]
            
            # Basic counts and volumes
            n_approved = len(approved_amounts)
            loan_volume = np.sum(approved_amounts)
            
            # Interest income (simplified: assume full term if no default)
            # For defaulted loans, assume they default halfway through term
            non_default_mask = approved_defaults == 0
            default_mask = approved_defaults == 1
            
            # Interest from non-defaulted loans (full term)
            full_term_interest = np.sum(
                approved_amounts[non_default_mask] * 
                approved_rates[non_default_mask] * 
                (loan_terms / 12.0)  # Convert to annual fraction
            )
            
            # Interest from defaulted loans (half term)
            partial_term_interest = np.sum(
                approved_amounts[default_mask] * 
                approved_rates[default_mask] * 
                (loan_terms / 24.0)  # Half term
            )
            
            interest_income = full_term_interest + partial_term_interest
            
            # Default losses (principal + accrued interest lost)
            default_principal_loss = np.sum(approved_amounts[default_mask])
            default_interest_loss = np.sum(
                approved_amounts[default_mask] * 
                approved_rates[default_mask] * 
                (loan_terms / 24.0)  # Remaining half term
            )
            default_losses = default_principal_loss + default_interest_loss
            
            # Net profit
            net_profit = interest_income - default_losses
            
            # Rates
            approval_rate = n_approved / len(approvals)
            default_rate = np.mean(approved_defaults) if n_approved > 0 else 0.0
            
            # Averages
            avg_loan_size = np.mean(approved_amounts)
            avg_interest_rate = np.mean(approved_rates)
            
            period_metrics = {
                'period': period,
                'n_approved': n_approved,
                'loan_volume': loan_volume,
                'interest_income': interest_income,
                'default_losses': default_losses,
                'net_profit': net_profit,
                'approval_rate': approval_rate,
                'default_rate': default_rate,
                'avg_loan_size': avg_loan_size,
                'avg_interest_rate': avg_interest_rate
            }
        
        # Update cumulative metrics
        self.total_loans_approved += period_metrics['n_approved']
        self.total_loan_volume += period_metrics['loan_volume']
        self.total_interest_income += period_metrics['interest_income']
        self.total_default_losses += period_metrics['default_losses']
        self.total_net_profit += period_metrics['net_profit']
        
        # Store period data
        self.period_data.append(period_metrics)
        
        return period_metrics
    
    def get_period_dataframe(self) -> pd.DataFrame:
        """Get period-by-period data as DataFrame.
        
        Returns:
            DataFrame with period accounting data
        """
        if not self.period_data:
            return pd.DataFrame()
        
        return pd.DataFrame(self.period_data)
    
    def get_cumulative_metrics(self) -> Dict[str, float]:
        """Get cumulative accounting metrics.
        
        Returns:
            Dictionary with cumulative metrics
        """
        return {
            'total_loans_approved': self.total_loans_approved,
            'total_loan_volume': self.total_loan_volume,
            'total_interest_income': self.total_interest_income,
            'total_default_losses': self.total_default_losses,
            'total_net_profit': self.total_net_profit,
            'overall_default_rate': self._compute_overall_default_rate(),
            'overall_profit_margin': self._compute_overall_profit_margin(),
            'return_on_loans': self._compute_return_on_loans()
        }
    
    def _compute_overall_default_rate(self) -> float:
        """Compute overall default rate across all periods."""
        if not self.period_data:
            return 0.0
        
        total_defaults = sum(
            p['n_approved'] * p['default_rate'] 
            for p in self.period_data
        )
        
        if self.total_loans_approved == 0:
            return 0.0
        
        return total_defaults / self.total_loans_approved
    
    def _compute_overall_profit_margin(self) -> float:
        """Compute overall profit margin."""
        if self.total_interest_income == 0:
            return 0.0
        
        return self.total_net_profit / self.total_interest_income
    
    def _compute_return_on_loans(self) -> float:
        """Compute return on total loan volume."""
        if self.total_loan_volume == 0:
            return 0.0
        
        return self.total_net_profit / self.total_loan_volume
    
    def reset(self) -> None:
        """Reset all accounting data."""
        self.total_loans_approved = 0
        self.total_loan_volume = 0.0
        self.total_interest_income = 0.0
        self.total_default_losses = 0.0
        self.total_net_profit = 0.0
        self.period_data = []
    
    def get_summary_report(self) -> str:
        """Get formatted summary report.
        
        Returns:
            Formatted accounting summary
        """
        if not self.period_data:
            return "No accounting data available."
        
        cumulative = self.get_cumulative_metrics()
        n_periods = len(self.period_data)
        
        report_lines = [
            "=== Bank Accounting Summary ===",
            f"Periods: {n_periods}",
            f"Total loans approved: {cumulative['total_loans_approved']:,}",
            f"Total loan volume: ${cumulative['total_loan_volume']:,.2f}",
            f"Total interest income: ${cumulative['total_interest_income']:,.2f}",
            f"Total default losses: ${cumulative['total_default_losses']:,.2f}",
            f"Total net profit: ${cumulative['total_net_profit']:,.2f}",
            f"Overall default rate: {cumulative['overall_default_rate']:.1%}",
            f"Overall profit margin: {cumulative['overall_profit_margin']:.1%}",
            f"Return on loans: {cumulative['return_on_loans']:.1%}",
        ]
        
        return "\n".join(report_lines)
