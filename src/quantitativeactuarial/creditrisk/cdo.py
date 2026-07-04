"""One-factor Gaussian copula CDO tranche valuation."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm


def gauss_hermite_normal(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return nodes and weights for integration under a standard normal."""
    x_nodes, raw_weights = np.polynomial.hermite.hermgauss(nodes)
    factor_nodes = x_nodes * np.sqrt(2)
    weights = raw_weights / np.sqrt(np.pi)
    return factor_nodes, weights


def log_binomial_coefficient(n: int, k: int) -> float:
    """Compute ``log(comb(n, k))`` using log-gamma values."""
    return float(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))


def conditional_default_probability(
    t: float, hazard_rate: float, rho: float, factor: np.ndarray
) -> np.ndarray:
    """
    Compute Hull one-factor conditional default probability.

    ``Q(t) = 1 - exp(-h t)``
    ``Q(t|F) = Phi((Phi^-1(Q(t)) - sqrt(rho) F) / sqrt(1-rho))``
    """
    unconditional = 1.0 - np.exp(-hazard_rate * t)
    inv_q = norm.ppf(unconditional)
    arg = (inv_q - np.sqrt(rho) * factor) / np.sqrt(1.0 - rho)
    return norm.cdf(arg)


def binomial_probabilities_log(n: int, q: np.ndarray, k_max: int) -> np.ndarray:
    """Return conditional binomial probabilities for ``k = 0..k_max-1``."""
    eps = 1e-300
    log_q = np.log(np.clip(q, eps, 1 - eps))
    log_1q = np.log(np.clip(1.0 - q, eps, 1 - eps))
    probs = np.zeros((k_max, len(q)))
    for k in range(k_max):
        log_p = log_binomial_coefficient(n, k) + k * log_q + (n - k) * log_1q
        probs[k] = np.exp(log_p)
    return probs


def expected_tranche_survival_given_factor(
    t: float,
    hazard_rate: float,
    rho: float,
    n: int,
    recovery_rate: float,
    attachment: float,
    detachment: float,
    factor_nodes: np.ndarray,
) -> np.ndarray:
    """Expected remaining tranche notional for each Gaussian factor node."""
    q = conditional_default_probability(t, hazard_rate, rho, factor_nodes)
    lgd = 1.0 - recovery_rate
    n_l = attachment * n / lgd
    n_h = detachment * n / lgd
    m_l = int(np.ceil(n_l))
    m_h = int(np.ceil(n_h))
    probs = binomial_probabilities_log(n, q, m_h)
    expected = probs[:m_l].sum(axis=0)
    for k in range(m_l, m_h):
        fraction = np.clip((detachment - k * lgd / n) / (detachment - attachment), 0.0, 1.0)
        expected += probs[k] * fraction
    return expected


def valuar_tranche(
    hazard_rate: float,
    rho: float,
    n: int,
    recovery_rate: float,
    attachment: float,
    detachment: float,
    risk_free_rate: float,
    periods: list[float] | tuple[float, ...] | np.ndarray,
    factor_nodes: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """
    Value a synthetic CDO tranche with Hull's one-factor Gaussian copula.

    Returns fee leg components ``A`` and ``B``, protection leg ``C``, fair
    spread, terminal expected tranche survival, and the factor-level survival
    matrix.
    """
    taus = [0.0] + list(periods)
    n_nodes = len(factor_nodes)
    expected_survival = np.zeros((len(taus), n_nodes))
    expected_survival[0] = 1.0

    for j, t in enumerate(periods):
        expected_survival[j + 1] = expected_tranche_survival_given_factor(
            t, hazard_rate, rho, n, recovery_rate, attachment, detachment, factor_nodes
        )

    fee_current = np.zeros(n_nodes)
    fee_accrued = np.zeros(n_nodes)
    protection = np.zeros(n_nodes)

    for j in range(1, len(taus)):
        tau_j = taus[j]
        tau_prev = taus[j - 1]
        delta = tau_j - tau_prev
        tau_mid = 0.5 * (tau_j + tau_prev)
        discount_j = np.exp(-risk_free_rate * tau_j)
        discount_mid = np.exp(-risk_free_rate * tau_mid)
        surv_j = expected_survival[j]
        surv_prev = expected_survival[j - 1]
        loss_increment = surv_prev - surv_j

        fee_current += delta * surv_j * discount_j
        fee_accrued += 0.5 * delta * loss_increment * discount_mid
        protection += loss_increment * discount_mid

    a = float((weights * fee_current).sum())
    b = float((weights * fee_accrued).sum())
    c = float((weights * protection).sum())
    spread = c / (a + b) if (a + b) > 1e-15 else np.nan
    terminal_survival = float((weights * expected_survival[-1]).sum())

    return {
        "A": a,
        "B": b,
        "C": c,
        "spread": float(spread),
        "spread_bps": float(spread * 10_000),
        "E_T": terminal_survival,
        "E_all": expected_survival,
    }


__all__ = [
    "gauss_hermite_normal",
    "log_binomial_coefficient",
    "conditional_default_probability",
    "binomial_probabilities_log",
    "expected_tranche_survival_given_factor",
    "valuar_tranche",
]
