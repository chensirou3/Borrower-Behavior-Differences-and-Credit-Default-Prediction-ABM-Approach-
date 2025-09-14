"""Configuration schema using Pydantic."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class TraitDistConfig(BaseModel):
    """Configuration for a single trait distribution."""
    
    mean: float = Field(..., description="Mean of the distribution")
    sd: float = Field(..., gt=0, description="Standard deviation")
    min: Optional[float] = Field(None, description="Minimum value (truncation)")
    max: Optional[float] = Field(None, description="Maximum value (truncation)")


class TraitsConfig(BaseModel):
    """Configuration for trait sampling."""
    
    gamma: TraitDistConfig = Field(
        default=TraitDistConfig(mean=2.0, sd=0.6, min=0.5, max=None),
        description="Risk appetite trait"
    )
    beta: TraitDistConfig = Field(
        default=TraitDistConfig(mean=0.90, sd=0.08, min=0.60, max=1.00),
        description="Financial discipline trait"
    )
    kappa: TraitDistConfig = Field(
        default=TraitDistConfig(mean=0.50, sd=0.25, min=0.00, max=1.50),
        description="Behavioral volatility trait"
    )
    omega: TraitDistConfig = Field(
        default=TraitDistConfig(mean=0.00, sd=0.80, min=None, max=None),
        description="External shock sensitivity trait"
    )
    eta: TraitDistConfig = Field(
        default=TraitDistConfig(mean=0.70, sd=0.20, min=0.00, max=1.00),
        description="Learning/adaptation trait"
    )


class ProxyMappingConfig(BaseModel):
    """Configuration for a single proxy mapping."""
    
    kappa: float = Field(default=0.0, description="Coefficient for kappa trait")
    beta: float = Field(default=0.0, description="Coefficient for beta trait")
    gamma: float = Field(default=0.0, description="Coefficient for gamma trait")
    omega: float = Field(default=0.0, description="Coefficient for omega trait")
    eta: float = Field(default=0.0, description="Coefficient for eta trait")
    intercept: float = Field(default=0.0, description="Intercept term")
    clip: Optional[List[float]] = Field(None, description="Clipping bounds [min, max]")
    min: Optional[float] = Field(None, description="Minimum value constraint")


class ProxiesConfig(BaseModel):
    """Configuration for proxy mappings."""
    
    noise_sd: float = Field(default=0.12, gt=0, description="Noise standard deviation")
    mapping: Dict[str, ProxyMappingConfig] = Field(
        default_factory=lambda: {
            "night_active_ratio": ProxyMappingConfig(
                kappa=0.50, beta=-0.20, intercept=0.20, clip=[0.0, 1.0]
            ),
            "session_std": ProxyMappingConfig(
                kappa=0.80, intercept=0.50, min=0.01
            ),
            "task_completion_ratio": ProxyMappingConfig(
                kappa=-0.40, beta=-0.20, intercept=0.85, clip=[0.0, 1.0]
            ),
            "spending_volatility": ProxyMappingConfig(
                kappa=0.50, beta=-0.20, omega=0.30, intercept=0.30, min=0.01
            ),
        },
        description="Mapping configurations for each proxy"
    )


class SineEnvConfig(BaseModel):
    """Configuration for sine wave environment."""
    
    enabled: bool = Field(default=True, description="Enable sine environment")
    period: int = Field(default=120, gt=0, description="Period in months")
    ar1_rho: float = Field(default=0.2, ge=0, le=1, description="AR(1) coefficient")
    noise_sd: float = Field(default=0.05, gt=0, description="Noise standard deviation")


class InterestConfig(BaseModel):
    """Configuration for interest rates."""
    
    r_mid_annual: float = Field(default=0.12, gt=0, description="Mid-point annual rate")
    r_amp_annual: float = Field(default=0.06, gt=0, description="Amplitude annual rate")


class ApprovalConfig(BaseModel):
    """Configuration for approval rates."""
    
    q_mid: float = Field(default=0.70, gt=0, le=1, description="Mid-point approval rate")
    q_amp: float = Field(default=0.15, gt=0, description="Approval rate amplitude")


class MacroNegConfig(BaseModel):
    """Configuration for macro negative indicators."""
    
    m0: float = Field(default=0.10, ge=0, description="Base macro negative level")
    m1: float = Field(default=0.25, ge=0, description="Macro negative amplitude")


class EnvironmentConfig(BaseModel):
    """Configuration for environment cycles."""
    
    sine: SineEnvConfig = Field(default_factory=SineEnvConfig)
    interest: InterestConfig = Field(default_factory=InterestConfig)
    approval: ApprovalConfig = Field(default_factory=ApprovalConfig)
    macro_neg: MacroNegConfig = Field(default_factory=MacroNegConfig)


class ApplicationConfig(BaseModel):
    """Configuration for loan applications."""
    
    base_rate: float = Field(default=0.30, gt=0, le=1, description="Base application rate")
    amp_with_env: float = Field(default=0.05, ge=0, description="Environment amplitude")


class LoanConfig(BaseModel):
    """Configuration for loan amounts."""
    
    base_multiple_month_income: float = Field(
        default=0.30, gt=0, description="Base multiple of monthly income"
    )
    noise_sd: float = Field(default=2000.0, gt=0, description="Noise standard deviation")
    min_amount: float = Field(default=500.0, gt=0, description="Minimum loan amount")
    max_amount: float = Field(default=50000.0, gt=0, description="Maximum loan amount")


class DGPCoefsConfig(BaseModel):
    """Configuration for DGP logit coefficients."""
    
    a0: float = Field(default=-2.0, description="Intercept")
    a1_dti: float = Field(default=3.2, description="DTI coefficient")
    a2_macro_neg: float = Field(default=1.5, description="Macro negative coefficient")
    a3_one_minus_beta: float = Field(default=1.3, description="(1-beta) coefficient")
    a4_kappa: float = Field(default=1.1, description="Kappa coefficient")
    a5_gamma: float = Field(default=0.2, description="Gamma coefficient")
    a6_rate_m: float = Field(default=0.8, description="Monthly rate coefficient")
    a7_prior_default: float = Field(default=0.9, description="Prior default coefficient")


class DGPConfig(BaseModel):
    """Configuration for data generation process."""
    
    logit_coefs: DGPCoefsConfig = Field(default_factory=DGPCoefsConfig)
    target_default_rate: List[float] = Field(
        default=[0.08, 0.15], description="Target default rate range"
    )


class PricingConfig(BaseModel):
    """Configuration for loan pricing."""
    
    r0_spread_annual: float = Field(default=0.00, ge=0, description="Base spread")
    phi_per_pd: float = Field(default=0.00, ge=0, description="Risk premium per PD")


class CapitalConfig(BaseModel):
    """Configuration for capital constraints."""
    
    enabled: bool = Field(default=False, description="Enable capital constraints")
    max_bad_rate: float = Field(default=0.12, gt=0, le=1, description="Max bad rate")


class BankPolicyConfig(BaseModel):
    """Configuration for bank policies."""
    
    decision_mode: str = Field(default="cap", description="Decision mode: cap or threshold")
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    capital: CapitalConfig = Field(default_factory=CapitalConfig)


class FeaturesConfig(BaseModel):
    """Configuration for feature sets."""
    
    baseline: List[str] = Field(
        default=["dti", "income_m", "rate_m", "macro_neg", "prior_defaults", "loan"],
        description="Baseline features"
    )
    proxies: List[str] = Field(
        default=["night_active_ratio", "session_std", "task_completion_ratio", "spending_volatility"],
        description="Proxy features"
    )


class AlgorithmConfig(BaseModel):
    """Configuration for a single algorithm."""
    
    name: str = Field(..., description="Algorithm name")
    calibrate: str = Field(default="none", description="Calibration method")
    params: Dict[str, Any] = Field(default_factory=dict, description="Algorithm parameters")


class SplitConfig(BaseModel):
    """Configuration for train/test split."""
    
    mode: str = Field(default="holdout", description="Split mode: holdout or oot")
    test_size: float = Field(default=0.30, gt=0, lt=1, description="Test size fraction")


class ModelingConfig(BaseModel):
    """Configuration for modeling."""
    
    algorithms: List[AlgorithmConfig] = Field(
        default_factory=lambda: [
            AlgorithmConfig(name="logistic", calibrate="none"),
            AlgorithmConfig(
                name="xgboost",
                params={
                    "n_estimators": 200,
                    "max_depth": 3,
                    "learning_rate": 0.08,
                    "subsample": 0.9,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 1.0,
                }
            ),
        ]
    )
    split: SplitConfig = Field(default_factory=SplitConfig)


class FairnessConfig(BaseModel):
    """Configuration for fairness evaluation."""
    
    group_by: str = Field(default="night_active_high", description="Grouping variable")
    eo_tpr_gap: bool = Field(default=True, description="Calculate EO/TPR gap")


class PlotsConfig(BaseModel):
    """Configuration for plots."""
    
    roc: bool = Field(default=True, description="Generate ROC curves")
    pr: bool = Field(default=True, description="Generate PR curves")
    ks: bool = Field(default=True, description="Generate KS plots")
    calibration: bool = Field(default=True, description="Generate calibration plots")


class EvaluationConfig(BaseModel):
    """Configuration for evaluation."""
    
    fairness: FairnessConfig = Field(default_factory=FairnessConfig)
    plots: PlotsConfig = Field(default_factory=PlotsConfig)


class OutputsConfig(BaseModel):
    """Configuration for outputs."""
    
    root_dir: str = Field(default="outputs", description="Root output directory")


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    
    level: str = Field(default="INFO", description="Logging level")


class QualityAssuranceConfig(BaseModel):
    """Configuration for quality assurance."""
    
    enabled: bool = Field(default=True, description="Enable quality assurance")
    prediction_validation: bool = Field(default=True, description="Validate predictions")
    monotonicity_checks: bool = Field(default=True, description="Check monotonicity")
    advantage_validation: bool = Field(default=True, description="Validate augmented advantage")
    regime_validation: bool = Field(default=True, description="Validate regime performance")
    tolerance: float = Field(default=0.002, description="Numerical tolerance")
    min_advantage_ratio: float = Field(default=0.8, description="Minimum advantage ratio")


class PopulationConfig(BaseModel):
    """Configuration for population."""
    
    N: int = Field(default=5000, gt=0, description="Population size")


class TimelineConfig(BaseModel):
    """Configuration for timeline."""
    
    T: int = Field(default=120, gt=0, description="Number of time periods")
    start_period: int = Field(default=1, ge=1, description="Starting period")


class Config(BaseModel):
    """Main configuration schema."""
    
    seed: int = Field(default=42, description="Random seed")
    population: PopulationConfig = Field(default_factory=PopulationConfig)
    timeline: TimelineConfig = Field(default_factory=TimelineConfig)
    traits: TraitsConfig = Field(default_factory=TraitsConfig)
    proxies: ProxiesConfig = Field(default_factory=ProxiesConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    application: ApplicationConfig = Field(default_factory=ApplicationConfig)
    loan: LoanConfig = Field(default_factory=LoanConfig)
    dgp: DGPConfig = Field(default_factory=DGPConfig)
    bank_policy: BankPolicyConfig = Field(default_factory=BankPolicyConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    modeling: ModelingConfig = Field(default_factory=ModelingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    quality_assurance: QualityAssuranceConfig = Field(default_factory=QualityAssuranceConfig)

    class Config:
        """Pydantic configuration."""
        
        extra = "forbid"  # Forbid extra fields
        validate_assignment = True  # Validate on assignment
