"""Advanced derivatives valuation models.

This subpackage contains pure pricing models, numerical methods, calibration
workflows, simulations, and analytics migrated from the VQD Streamlit app.
"""

from .analytics.distributions import distribution_moments, excess_kurtosis, theoretical_mc_error
from .analytics.parity import forward_price_continuous, intrinsic_time_value, parity_check
from .analytics.volatility import iv_rv_spread, realized_vol
from .calibration.heston import calibrate_heston
from .calibration.sabr import calibrate_sabr
from .comparison import compare_all_models
from .models.bachelier import BachelierEngine
from .models.binomial import CRREngine, crr_tree_parameters
from .models.black_scholes import BSMEngine, bsm_d1_d2_summary
from .models.heston import HestonEngine
from .models.merton import MertonEngine, merton_jump_statistics
from .models.nig import NIGEngine
from .models.sabr import SABREngine
from .models.variance_gamma import VarianceGammaEngine
from .monte_carlo import monte_carlo_bsm
from .simulation.gbm import simulate_gbm_paths
from .simulation.levy import simulate_vg_nig_returns
from .simulation.merton import simulate_merton_paths

__all__ = [
    "BSMEngine",
    "BachelierEngine",
    "CRREngine",
    "HestonEngine",
    "MertonEngine",
    "NIGEngine",
    "SABREngine",
    "VarianceGammaEngine",
    "bsm_d1_d2_summary",
    "calibrate_heston",
    "calibrate_sabr",
    "compare_all_models",
    "crr_tree_parameters",
    "distribution_moments",
    "excess_kurtosis",
    "forward_price_continuous",
    "intrinsic_time_value",
    "iv_rv_spread",
    "merton_jump_statistics",
    "monte_carlo_bsm",
    "parity_check",
    "realized_vol",
    "simulate_gbm_paths",
    "simulate_merton_paths",
    "simulate_vg_nig_returns",
    "theoretical_mc_error",
]
