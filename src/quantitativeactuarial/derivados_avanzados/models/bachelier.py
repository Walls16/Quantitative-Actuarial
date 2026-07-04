"""Bachelier normal-volatility option pricing."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..numerics.implied_volatility import implied_normal_volatility
from .black_scholes import BSMEngine


class BachelierEngine:
    """
    Bachelier (1900) normal model — assumes arithmetic Brownian motion.
    dS = sigma_n * dW  (absolute, not percentage)

    Used in: interest rate options, inflation derivatives, negative-rate environments.
    sigma_n is the *normal* vol (in price units, not percentage).

    Parameters
    ----------
    S, K, T, r, q : standard
    sigma_n        : normal volatility (absolute, e.g. $2 per year)
    """

    def __init__(self, S, K, T, r, sigma_n, q=0.0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma_n = sigma_n
        self.q = q

    def _d(self):
        if self.T <= 0 or self.sigma_n <= 0:
            return 0.0
        return (self.S * np.exp(-self.q * self.T) - self.K * np.exp(-self.r * self.T)) / (
            self.sigma_n * np.sqrt(self.T)
        )

    def call_price(self):
        """Bachelier call: C = e^{-rT}[(F-K)N(d) + sigma_n*sqrt(T)*n(d)]"""
        if self.T <= 0:
            return max(self.S - self.K, 0.0)
        F = self.S * np.exp((self.r - self.q) * self.T)
        vol = self.sigma_n * np.sqrt(self.T)
        d = (F - self.K) / vol
        return np.exp(-self.r * self.T) * ((F - self.K) * norm.cdf(d) + vol * norm.pdf(d))

    def put_price(self):
        """Bachelier put: P = e^{-rT}[(K-F)N(-d) + sigma_n*sqrt(T)*n(d)]"""
        if self.T <= 0:
            return max(self.K - self.S, 0.0)
        F = self.S * np.exp((self.r - self.q) * self.T)
        vol = self.sigma_n * np.sqrt(self.T)
        d = (F - self.K) / vol
        return np.exp(-self.r * self.T) * ((self.K - F) * norm.cdf(-d) + vol * norm.pdf(d))

    def greeks(self):
        """Analytical Bachelier Greeks."""
        F = self.S * np.exp((self.r - self.q) * self.T)
        vol = self.sigma_n * np.sqrt(self.T)
        d = (F - self.K) / vol if vol > 0 else 0.0
        disc = np.exp(-self.r * self.T)

        call_delta = disc * norm.cdf(d)
        put_delta = disc * (norm.cdf(d) - 1)
        gamma = disc * norm.pdf(d) / (self.sigma_n * np.sqrt(self.T)) if self.T > 0 else 0.0
        vega = disc * np.sqrt(self.T) * norm.pdf(d)  # per unit sigma_n
        call_theta = (
            (
                -disc * self.sigma_n * norm.pdf(d) / (2 * np.sqrt(self.T))
                - self.r * self.call_price()
            )
            / 365
            if self.T > 0
            else 0.0
        )

        return {
            "call_delta": call_delta,
            "put_delta": put_delta,
            "gamma": gamma,
            "vega": vega,
            "call_theta": call_theta,
        }

    def implied_normal_vol(self, market_price, option_type="call", tol=1e-8):
        """Invert Bachelier formula to get normal vol from market price."""
        return implied_normal_volatility(
            market_price, self.S, self.K, self.T, self.r, self.q, option_type, tol
        )

    def lognormal_vol(self):
        """
        Approximate equivalent BSM lognormal vol via Hagan formula:
        sigma_LN ≈ sigma_N / F  for ATM.
        More accurate via Bachelier-BSM parity at the given strike.
        """
        bsm = BSMEngine(self.S, self.K, self.T, self.r, 0.3, self.q)
        price = self.call_price()
        return bsm.implied_vol(price, "call")

    def vol_smile(self, strikes):
        """
        Bachelier IV smile — in a pure normal model this is FLAT.
        Useful to see vs lognormal smile.
        """
        return np.array([self.sigma_n for _ in strikes])
