"""Flat credit-risk API."""

from .cdo import (
    binomial_probabilities_log,
    conditional_default_probability,
    expected_tranche_survival_given_factor,
    gauss_hermite_normal,
    log_binomial_coefficient,
    value_tranche,
)
from .cds import (
    cds_accrued_premium_table,
    cds_contingent_leg_table,
    cds_fair_spread,
    cds_premium_leg_table,
    cds_probability_table,
    value_cds,
)
from .copula import gaussian_copula_simulation, thresholds_per_bond
from .data import (
    DEFAULT_SPREADS,
    DEFAULT_TREASURY,
    NR_METHODS,
    N_DEST,
    N_EMIT,
    RATINGS_DEST,
    RATINGS_EMIT,
    RATINGS_VAL,
    RATING_IDX,
    TRADING_DAYS,
    _TM_RAW_17x19,
)
from .distribution import expected_value_and_sigma, independent_distribution
from .risk import (
    scale_var_cvar,
    var_cvar_from_distribution,
    var_cvar_from_simulations,
    var_cvar_parametric,
)
from .transitions import DEFAULT_TM, RATINGS_NO_D, _TM_SIZE_BY_MODE, build_transition_matrix
from .types import BondData, Distribution, RiskResult, RiskResults
from .valuation import bond_values_per_rating

__all__ = [
    "RATINGS_DEST",
    "RATINGS_EMIT",
    "RATINGS_VAL",
    "N_EMIT",
    "N_DEST",
    "RATING_IDX",
    "TRADING_DAYS",
    "_TM_RAW_17x19",
    "DEFAULT_SPREADS",
    "DEFAULT_TREASURY",
    "NR_METHODS",
    "DEFAULT_TM",
    "_TM_SIZE_BY_MODE",
    "RATINGS_NO_D",
    "build_transition_matrix",
    "cds_probability_table",
    "cds_premium_leg_table",
    "cds_contingent_leg_table",
    "cds_accrued_premium_table",
    "cds_fair_spread",
    "value_cds",
    "gauss_hermite_normal",
    "log_binomial_coefficient",
    "conditional_default_probability",
    "binomial_probabilities_log",
    "expected_tranche_survival_given_factor",
    "value_tranche",
    "bond_values_per_rating",
    "independent_distribution",
    "expected_value_and_sigma",
    "thresholds_per_bond",
    "gaussian_copula_simulation",
    "var_cvar_from_distribution",
    "scale_var_cvar",
    "var_cvar_from_simulations",
    "var_cvar_parametric",
    "BondData",
    "RiskResult",
    "RiskResults",
    "Distribution",
]
