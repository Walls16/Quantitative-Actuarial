"""Cox-Ross-Rubinstein binomial tree pricing."""

from __future__ import annotations

import numpy as np


class CRREngine:
    """Cox-Ross-Rubinstein binomial tree."""

    def __init__(self, S, K, T, r, sigma, q=0.0, N=200, american=False):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.N = N
        self.american = american

    def _tree_params(self):
        dt = self.T / self.N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((self.r - self.q) * dt) - d) / (u - d)
        disc = np.exp(-self.r * dt)
        return dt, u, d, p, disc

    def price(self, option_type="call"):
        dt, u, d, p, disc = self._tree_params()
        N = self.N

        # Terminal stock prices
        ST = self.S * (u ** (np.arange(N, -1, -1))) * (d ** (np.arange(0, N + 1, 1)))

        # Terminal payoffs
        if option_type == "call":
            V = np.maximum(ST - self.K, 0)
        else:
            V = np.maximum(self.K - ST, 0)

        # Backward induction
        for _ in range(N):
            V = disc * (p * V[:-1] + (1 - p) * V[1:])
            if self.american:
                ST = ST[:-1] / u  # one step back
                if option_type == "call":
                    V = np.maximum(V, np.maximum(ST - self.K, 0))
                else:
                    V = np.maximum(V, np.maximum(self.K - ST, 0))

        return float(V[0])

    def call_price(self):
        return self.price("call")

    def put_price(self):
        return self.price("put")

    def full_tree(self, option_type="call", max_display=7):
        """Returns stock and option trees truncated to max_display steps for visualization."""
        dt, u, d, p, disc = self._tree_params()
        N = min(self.N, max_display)

        S_tree = [[0.0] * (i + 1) for i in range(N + 1)]
        for i in range(N + 1):
            for j in range(i + 1):
                S_tree[i][j] = self.S * (u ** (i - j)) * (d**j)

        # Terminal payoffs
        if option_type == "call":
            V_tree = [max(s - self.K, 0) for s in S_tree[N]]
        else:
            V_tree = [max(self.K - s, 0) for s in S_tree[N]]

        V_full = [None] * N + [V_tree]
        for i in range(N - 1, -1, -1):
            row = []
            for j in range(i + 1):
                val = disc * (p * V_full[i + 1][j] + (1 - p) * V_full[i + 1][j + 1])
                if self.american:
                    s = S_tree[i][j]
                    intrinsic = max(s - self.K, 0) if option_type == "call" else max(self.K - s, 0)
                    val = max(val, intrinsic)
                row.append(val)
            V_full[i] = row

        return S_tree, V_full

    def delta(self, option_type="call"):
        dt, u, d, p, disc = self._tree_params()
        Su = self.S * u
        Sd = self.S * d
        eng_u = CRREngine(
            Su, self.K, self.T - dt, self.r, self.sigma, self.q, self.N - 1, self.american
        )
        eng_d = CRREngine(
            Sd, self.K, self.T - dt, self.r, self.sigma, self.q, self.N - 1, self.american
        )
        fu = eng_u.price(option_type)
        fd = eng_d.price(option_type)
        return (fu - fd) / (Su - Sd)


def crr_tree_parameters(
    T: float, r: float, sigma: float, q: float = 0.0, N: int = 200
) -> dict[str, float]:
    """Return Cox-Ross-Rubinstein one-step tree parameters.

    The up/down factors are ``u = exp(sigma sqrt(dt))`` and ``d = 1/u`` with
    ``dt = T/N``.  The risk-neutral probability is
    ``p = [exp((r-q)dt) - d] / (u - d)`` and the one-step discount factor is
    ``exp(-r dt)``.
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    return {"dt": float(dt), "u": float(u), "d": float(d), "p": float(p), "disc": float(disc)}
