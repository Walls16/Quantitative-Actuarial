"""Normal Inverse Gaussian model pricing."""

from __future__ import annotations

import numpy as np

from ..numerics.carr_madan_fft import carr_madan_option_price
from ..numerics.implied_volatility import implied_volatility_black_scholes


class NIGEngine:
    """Normal Inverse Gaussian model via characteristic function and Carr-Madan FFT."""

    def __init__(self, S, K, T, r, alpha, beta, delta, q=0.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if abs(beta) >= alpha:
            raise ValueError("beta must satisfy abs(beta) < alpha.")
        if beta + 1 >= alpha:
            raise ValueError(
                "alpha must be greater than beta + 1 for the risk-neutral drift adjustment."
            )
        if delta <= 0:
            raise ValueError("delta must be positive.")
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.q = q

    def _mu(self):
        a, b, d = self.alpha, self.beta, self.delta
        return d * (np.sqrt(a**2 - b**2) - np.sqrt(a**2 - (b + 1) ** 2))

    def characteristic_function(self, u):
        a, b, d = self.alpha, self.beta, self.delta
        mu = self._mu()
        drift = 1j * u * (np.log(self.S) + (self.r - self.q + mu) * self.T)
        nig = -d * self.T * (np.sqrt(a**2 - (b + 1j * u) ** 2) - np.sqrt(a**2 - b**2))
        return np.exp(drift + nig)

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
            eng = NIGEngine(
                self.S, strike, self.T, self.r, self.alpha, self.beta, self.delta, self.q
            )
            iv = eng.implied_vol(option_type)
            ivs.append(iv * 100 if not np.isnan(iv) else np.nan)
        return np.array(ivs)
