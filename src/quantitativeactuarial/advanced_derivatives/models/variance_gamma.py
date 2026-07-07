"""Variance Gamma model pricing."""

from __future__ import annotations

import numpy as np

from ..numerics.carr_madan_fft import carr_madan_option_price
from ..numerics.implied_volatility import implied_volatility_black_scholes


class VarianceGammaEngine:
    """Variance Gamma model (Madan, Carr & Chang 1998)."""

    def __init__(self, S, K, T, r, sigma, theta, nu, q=0.0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.theta = theta
        self.nu = nu
        self.q = q

    def _omega(self):
        """Risk-neutral drift correction."""
        return (1 / self.nu) * np.log(1 - self.theta * self.nu - 0.5 * self.sigma**2 * self.nu)

    def characteristic_function(self, u):
        """Characteristic function of log(S_T) under the risk-neutral measure."""
        omega = self._omega()
        drift_term = 1j * u * (np.log(self.S) + (self.r - self.q + omega) * self.T)
        vg_term = -(self.T / self.nu) * np.log(
            1 - 1j * u * self.theta * self.nu + 0.5 * self.sigma**2 * self.nu * u**2
        )
        return np.exp(drift_term + vg_term)

    def _char_func(self, u):
        return self.characteristic_function(u)

    def _price_via_fft(self, option_type="call", N=4096, eta=0.25):
        return carr_madan_option_price(
            self.characteristic_function,
            self.S,
            self.K,
            self.T,
            self.r,
            self.q,
            option_type,
            n_grid=N,
            eta=eta,
        )

    def call_price(self):
        return self._price_via_fft("call")

    def put_price(self):
        return self._price_via_fft("put")

    def implied_vol(self, option_type="call"):
        price = self.call_price() if option_type == "call" else self.put_price()
        return implied_volatility_black_scholes(
            price, self.S, self.K, self.T, self.r, self.q, option_type
        )

    def vol_smile(self, strikes, option_type="call"):
        ivs = []
        for strike in strikes:
            eng = VarianceGammaEngine(
                self.S, strike, self.T, self.r, self.sigma, self.theta, self.nu, self.q
            )
            iv = eng.implied_vol(option_type)
            ivs.append(iv * 100 if not np.isnan(iv) else np.nan)
        return np.array(ivs)
