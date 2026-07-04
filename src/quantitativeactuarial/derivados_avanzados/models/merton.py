"""Merton jump-diffusion option pricing."""

from __future__ import annotations

import math

import numpy as np

from .black_scholes import BSMEngine


class MertonEngine:
    """
    Merton (1976) jump-diffusion model.
    Price is a Poisson-weighted sum of BSM prices with adjusted parameters.
    """

    def __init__(self, S, K, T, r, sigma, q=0.0, lam=1.0, mu_j=0.0, sigma_j=0.2, n_terms=50):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.lam = lam  # jump intensity (jumps per year)
        self.mu_j = mu_j  # mean log-jump size
        self.sigma_j = sigma_j  # jump vol
        self.n_terms = n_terms

    def _kappa(self):
        """Expected proportional jump size."""
        return np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1

    def price(self, option_type="call"):
        kappa = self._kappa()
        lam_prime = self.lam * (1 + kappa)
        total = 0.0

        for n in range(self.n_terms):
            # Poisson weight under λ' = λ(1+κ)
            poisson_w = np.exp(-lam_prime * self.T) * (lam_prime * self.T) ** n / math.factorial(n)
            if poisson_w < 1e-15:
                break

            # Adjusted vol for n jumps
            sigma_n = np.sqrt(self.sigma**2 + n * self.sigma_j**2 / self.T)

            # Adjusted rate: compensate for jump drift so risk-neutral condition holds
            r_n = self.r - self.lam * kappa + n * (self.mu_j + 0.5 * self.sigma_j**2) / self.T

            bsm = BSMEngine(self.S, self.K, self.T, r_n, sigma_n, self.q)
            price_n = bsm.call_price() if option_type == "call" else bsm.put_price()
            total += poisson_w * price_n

        return total

    def call_price(self):
        return self.price("call")

    def put_price(self):
        return self.price("put")

    def vol_smile(self, strikes):
        """Jump-diffusion implied vol smile."""
        ivs = []
        for K in strikes:
            eng_m = MertonEngine(
                self.S,
                K,
                self.T,
                self.r,
                self.sigma,
                self.q,
                self.lam,
                self.mu_j,
                self.sigma_j,
                self.n_terms,
            )
            price = eng_m.call_price()
            bsm = BSMEngine(self.S, K, self.T, self.r, 0.3, self.q)
            iv = bsm.implied_vol(price, "call")
            ivs.append(iv)
        return np.array(ivs)


def merton_jump_statistics(
    lam: float, mu_j: float, sigma_j: float, sigma: float
) -> dict[str, float]:
    """Return jump-compensator statistics for the Merton jump-diffusion model.

    ``kappa_j = E[J - 1] = exp(mu_j + sigma_j^2/2) - 1`` is the expected
    proportional jump size used in the risk-neutral drift compensation.  The
    displayed total volatility proxy is ``sqrt(sigma^2 + lambda sigma_j^2)``.
    """
    kappa_j = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    return {
        "kappa_j": float(kappa_j),
        "mean_jump_pct": float(100.0 * kappa_j),
        "lambda_adjusted": float(lam * (1.0 + kappa_j)),
        "bsm_total_sigma": float(np.sqrt(sigma**2 + lam * sigma_j**2)),
    }
