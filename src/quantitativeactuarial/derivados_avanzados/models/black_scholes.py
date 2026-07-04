"""Black-Scholes-Merton analytical option pricing."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..numerics.implied_volatility import implied_volatility_black_scholes


class BSMEngine:
    """Black-Scholes-Merton analytical pricing."""

    def __init__(self, S, K, T, r, sigma, q=0.0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q

    def _d1(self):
        if self.T <= 0 or self.sigma <= 0:
            return 0.0
        return (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )

    def _d2(self):
        return self._d1() - self.sigma * np.sqrt(self.T)

    def call_price(self):
        if self.T <= 0:
            return max(self.S - self.K, 0.0)
        d1, d2 = self._d1(), self._d2()
        return self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(
            -self.r * self.T
        ) * norm.cdf(d2)

    def put_price(self):
        if self.T <= 0:
            return max(self.K - self.S, 0.0)
        d1, d2 = self._d1(), self._d2()
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * np.exp(
            -self.q * self.T
        ) * norm.cdf(-d1)

    def greeks(self):
        """Returns dict of all first and second-order Greeks."""
        d1, d2 = self._d1(), self._d2()
        sqrtT = np.sqrt(self.T) if self.T > 0 else 1e-10
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)

        discount = np.exp(-self.r * self.T)
        div_disc = np.exp(-self.q * self.T)

        #  Delta
        call_delta = div_disc * cdf_d1
        put_delta = div_disc * (cdf_d1 - 1)

        #  Gamma
        gamma = div_disc * pdf_d1 / (self.S * self.sigma * sqrtT)

        #  Theta (per calendar day)
        call_theta = (
            -(self.S * div_disc * pdf_d1 * self.sigma) / (2 * sqrtT)
            - self.r * self.K * discount * cdf_d2
            + self.q * self.S * div_disc * cdf_d1
        ) / 365

        put_theta = (
            -(self.S * div_disc * pdf_d1 * self.sigma) / (2 * sqrtT)
            + self.r * self.K * discount * norm.cdf(-d2)
            - self.q * self.S * div_disc * norm.cdf(-d1)
        ) / 365

        #  Vega (per 1% move in vol)
        vega = self.S * div_disc * pdf_d1 * sqrtT / 100

        #  Rho (per 1% move in rate)
        call_rho = self.K * self.T * discount * cdf_d2 / 100
        put_rho = -self.K * self.T * discount * norm.cdf(-d2) / 100

        #  Vanna
        vanna = -div_disc * pdf_d1 * d2 / self.sigma

        #  Volga / Vomma
        volga = self.S * div_disc * pdf_d1 * sqrtT * d1 * d2 / self.sigma

        #  Charm (delta decay per day)
        call_charm = (
            div_disc
            * (pdf_d1 * (self.r - self.q) / (self.sigma * sqrtT) - d2 / (2 * self.T) * pdf_d1)
        ) / 365

        return {
            "call_delta": call_delta,
            "put_delta": put_delta,
            "gamma": gamma,
            "call_theta": call_theta,
            "put_theta": put_theta,
            "vega": vega,
            "call_rho": call_rho,
            "put_rho": put_rho,
            "vanna": vanna,
            "volga": volga,
            "call_charm": call_charm,
        }

    def implied_vol(self, market_price, option_type="call", tol=1e-6, max_iter=200):
        """Black-Scholes-Merton implied-volatility solver."""
        return implied_volatility_black_scholes(
            market_price,
            self.S,
            self.K,
            self.T,
            self.r,
            self.q,
            option_type,
            tol,
            max_iter,
        )

    def vol_smile_surface(self, strikes, maturities):
        """
        Returns a 2D array of implied vols for a given vol surface.
        Used in Page 5 smile visualization.
        (Assumes flat vol input; real usage feeds market prices.)
        """
        surface = np.full((len(maturities), len(strikes)), self.sigma)
        return surface


def bsm_d1_d2_summary(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> dict[str, float]:
    """Return the Black-Scholes-Merton ``d1`` and ``d2`` values.

    ``d1 = [ln(S/K) + (r - q + sigma^2 / 2)T] / [sigma sqrt(T)]`` and
    ``d2 = d1 - sigma sqrt(T)``.  If maturity or volatility is non-positive,
    both values are reported as ``0.0`` to match the package pricing engines.
    """
    if T <= 0 or sigma <= 0:
        return {"d1": 0.0, "d2": 0.0}
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return {"d1": float(d1), "d2": float(d2)}
