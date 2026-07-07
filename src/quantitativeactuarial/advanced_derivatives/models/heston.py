"""Heston stochastic-volatility pricing."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from ..numerics.implied_volatility import implied_volatility_black_scholes


class HestonEngine:
    """
    Heston (1993) stochastic volatility model via semi-analytical characteristic function.
    dS = (r-q)S dt + sqrt(v) S dW1
    dv = kappa*(theta - v) dt + xi*sqrt(v) dW2,  corr(dW1,dW2) = rho
    """

    def __init__(self, S, K, T, r, q, v0, kappa, theta, xi, rho):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.v0 = v0  # initial variance
        self.kappa = kappa  # mean reversion speed
        self.theta = theta  # long-run variance
        self.xi = xi  # vol of vol
        self.rho = rho  # correlation

    def _char_func(self, phi, j):
        """
        Heston characteristic function — Albrecher et al. (2007) stable formulation.
        Avoids the discontinuity in the complex log branch cut.
        """
        S, K, T = self.S, self.K, self.T
        r, q = self.r, self.q
        v0, kappa, theta, xi, rho = self.v0, self.kappa, self.theta, self.xi, self.rho

        i = complex(0, 1)

        if j == 1:
            u = 0.5
            b = kappa - rho * xi
        else:
            u = -0.5
            b = kappa

        a = kappa * theta
        x = np.log(S)
        ln_K = np.log(K)

        d = np.sqrt((rho * xi * i * phi - b) ** 2 - xi**2 * (2 * u * i * phi - phi**2))

        # Use the formulation that avoids the principal value discontinuity
        g2 = (b - rho * xi * i * phi - d) / (b - rho * xi * i * phi + d)

        exp_dT = np.exp(-d * T)

        C = (r - q) * i * phi * T + (a / xi**2) * (
            (b - rho * xi * i * phi - d) * T - 2.0 * np.log((1.0 - g2 * exp_dT) / (1.0 - g2))
        )
        D = ((b - rho * xi * i * phi - d) / xi**2) * ((1.0 - exp_dT) / (1.0 - g2 * exp_dT))

        return np.exp(C + D * v0 + i * phi * (x - ln_K))

    def _integrand(self, phi, j):
        return np.real(self._char_func(phi, j) / (complex(0, 1) * phi))

    def _Pj(self, j, upper=200, limit=500):
        integral, _ = quad(self._integrand, 1e-6, upper, args=(j,), limit=limit)
        return 0.5 + integral / np.pi

    def call_price(self):
        P1 = self._Pj(1)
        P2 = self._Pj(2)
        return self.S * np.exp(-self.q * self.T) * P1 - self.K * np.exp(-self.r * self.T) * P2

    def put_price(self):
        call = self.call_price()
        return call - self.S * np.exp(-self.q * self.T) + self.K * np.exp(-self.r * self.T)

    def vol_smile(self, strikes):
        """Compute implied vol smile across strikes using BSM inversion."""
        ivs = []
        for K in strikes:
            eng_h = HestonEngine(
                self.S,
                K,
                self.T,
                self.r,
                self.q,
                self.v0,
                self.kappa,
                self.theta,
                self.xi,
                self.rho,
            )
            price = eng_h.call_price()
            iv = implied_volatility_black_scholes(price, self.S, K, self.T, self.r, self.q, "call")
            ivs.append(iv)
        return np.array(ivs)
