"""SABR implied-volatility approximation and option pricing."""

from __future__ import annotations

import numpy as np

from .black_scholes import BSMEngine


class SABREngine:
    """
    SABR model (Hagan, Kumar, Lesniewski, Woodward 2002).
    Provides an analytical approximation for implied volatility,
    then prices via BSM with that IV.

    dF = sigma * F^beta * dW1
    dsigma = alpha * sigma * dW2
    corr(dW1, dW2) = rho

    Parameters
    ----------
    F      : forward price = S * exp((r-q)*T)
    K      : strike
    T      : time to expiry
    alpha  : initial vol (SABR sigma_0)
    beta   : elasticity parameter in [0,1]
             0 = normal (Bachelier-like), 1 = lognormal (BSM-like)
    rho    : correlation dF-dsigma
    nu     : vol of vol (SABR alpha)
    r, q   : for discounting / forward calculation
    S      : spot (optional, used to compute F if not given directly)
    """

    def __init__(self, F, K, T, alpha, beta, rho, nu, r=0.0, q=0.0, S=None):
        self.F = F
        self.K = K
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
        self.r = r
        self.q = q
        self.S = S if S is not None else F

    def implied_vol(self):
        """
        Hagan et al. (2002) lognormal implied vol approximation.
        Numerically stable for F ≈ K (ATM) via L'Hopital expansion.
        """
        F, K, T = self.F, self.K, self.T
        alpha, beta = self.alpha, self.beta
        rho, nu = self.rho, self.nu

        eps = 1e-8
        if T <= 0:
            return alpha

        FK_mid = (F * K) ** ((1 - beta) / 2)
        ln_FK = np.log(F / K) if abs(F - K) > eps else 0.0

        # Leading term
        A = alpha / (
            FK_mid * (1 + (1 - beta) ** 2 / 24 * ln_FK**2 + (1 - beta) ** 4 / 1920 * ln_FK**4)
        )

        # z and x(z) correction
        if abs(F - K) > eps:
            z = nu / alpha * FK_mid * ln_FK
            x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
            B = z / x_z if abs(x_z) > eps else 1.0
        else:
            B = 1.0

        # Time-correction term
        C = (
            1
            + (
                (1 - beta) ** 2 / 24 * alpha**2 / FK_mid**2
                + rho * beta * nu * alpha / (4 * FK_mid)
                + (2 - 3 * rho**2) / 24 * nu**2
            )
            * T
        )

        return A * B * C

    def call_price(self):
        iv = self.implied_vol()
        bsm = BSMEngine(self.S, self.K, self.T, self.r, iv, self.q)
        return bsm.call_price()

    def put_price(self):
        iv = self.implied_vol()
        bsm = BSMEngine(self.S, self.K, self.T, self.r, iv, self.q)
        return bsm.put_price()

    def vol_smile(self, strikes):
        """IV smile across strikes."""
        ivs = []
        for K in strikes:
            eng = SABREngine(
                self.F, K, self.T, self.alpha, self.beta, self.rho, self.nu, self.r, self.q, self.S
            )
            ivs.append(eng.implied_vol())
        return np.array(ivs)
