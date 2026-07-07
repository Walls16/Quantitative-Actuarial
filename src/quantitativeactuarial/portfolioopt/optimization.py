"""
optimization.py
================

Implementa 21 estrategias de optimización de portafolios:

* Equal Weight (1/N)
* Maximo Sharpe Ratio
* Minima Varianza (GMV)
* Paridad de Riesgo (ERC)
* Inverso a Volatilidad
* Inverso a Varianza
* Maxima Diversificacion
* Minimo CVaR
* Maximo Sortino
* Maximo Calmar
* Hierarchical Risk Parity (HRP)
* Maxima Decorrelacion
* Minimo CDaR
* Volatilidad Objetivo (10%)
* Maximo Retorno (Long-only)
* Minimo Skewness Negativo
* Kelly Fraction (fraccion de Kelly)
* Black-Litterman (prior de mercado, sin views)
* Minima Kurtosis de Cola
* Maximo Omega Ratio
* Maxima Entropia

Todas devuelven un vector de pesos w que suma 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .hierarchical import hierarchical_risk_parity
from .solvers import solve_slsqp_weights

TRADING_DAYS = 252


class PortfolioOptimizer:
    """Motor de optimizacion de portafolios sobre una matriz de retornos."""

    def __init__(
        self,
        returns: pd.DataFrame,
        rf: float = 0.0,
        allow_short: bool = False,
        periods: int = TRADING_DAYS,
    ):
        self.returns = returns.dropna()
        self.tickers = list(self.returns.columns)
        self.n = len(self.tickers)
        self.rf = rf
        self.periods = periods
        self.allow_short = allow_short

        self.mean_returns = self.returns.mean() * periods
        self.cov_matrix = self.returns.cov() * periods
        self.corr_matrix = self.returns.corr()

        if allow_short:
            self.min_w, self.max_w = -1.0, 1.0
        else:
            self.min_w, self.max_w = 0.0, 1.0

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------
    def _minimize(self, objective, bounds=None, constraints=None, x0=None) -> np.ndarray:
        return solve_slsqp_weights(
            objective,
            self.n,
            self.min_w,
            self.max_w,
            bounds=bounds,
            constraints=constraints,
            x0=x0,
        )

    # ------------------------------------------------------------------
    # 1. Equal Weight
    # ------------------------------------------------------------------
    def equal_weight(self) -> np.ndarray:
        return np.repeat(1.0 / self.n, self.n)

    # ------------------------------------------------------------------
    # 2. Maximo Sharpe
    # ------------------------------------------------------------------
    def max_sharpe(self) -> np.ndarray:
        mu, cov = self.mean_returns.values, self.cov_matrix.values

        def neg_sharpe(w):
            ret = w @ mu
            vol = np.sqrt(max(w @ cov @ w, 1e-12))
            return -(ret - self.rf) / vol

        return self._minimize(neg_sharpe)

    # ------------------------------------------------------------------
    # 3. Minima Varianza
    # ------------------------------------------------------------------
    def min_variance(self) -> np.ndarray:
        cov = self.cov_matrix.values
        return self._minimize(lambda w: w @ cov @ w)

    # ------------------------------------------------------------------
    # 4. Paridad de Riesgo (ERC)
    # ------------------------------------------------------------------
    def risk_parity(self) -> np.ndarray:
        cov = self.cov_matrix.values
        target = 1.0 / self.n

        def erc_objective(w):
            port_var = w @ cov @ w
            if port_var <= 1e-12:
                return 1e6
            contrib = w * (cov @ w) / port_var
            return np.sum((contrib - target) ** 2)

        bounds = tuple((1e-4, 1.0) for _ in range(self.n))
        return self._minimize(erc_objective, bounds=bounds)

    # ------------------------------------------------------------------
    # 5. Inverso a Volatilidad
    # ------------------------------------------------------------------
    def inverse_volatility(self) -> np.ndarray:
        vol = np.sqrt(np.diag(self.cov_matrix.values))
        vol = np.where(vol == 0, 1e-8, vol)
        inv = 1.0 / vol
        return inv / inv.sum()

    # ------------------------------------------------------------------
    # 6. Inverso a Varianza
    # ------------------------------------------------------------------
    def inverse_variance(self) -> np.ndarray:
        var = np.diag(self.cov_matrix.values)
        var = np.where(var == 0, 1e-8, var)
        inv = 1.0 / var
        return inv / inv.sum()

    # ------------------------------------------------------------------
    # 7. Maxima Diversificacion
    # ------------------------------------------------------------------
    def max_diversification(self) -> np.ndarray:
        sigma = np.sqrt(np.diag(self.cov_matrix.values))
        cov = self.cov_matrix.values

        def neg_div(w):
            port_vol = np.sqrt(max(w @ cov @ w, 1e-12))
            return -(w @ sigma) / port_vol

        return self._minimize(neg_div)

    # ------------------------------------------------------------------
    # 8. Minimo CVaR
    # ------------------------------------------------------------------
    def min_cvar(self, alpha: float = 0.05) -> np.ndarray:
        R = self.returns.values

        def cvar_obj(w):
            pr = R @ w
            var = np.quantile(pr, alpha)
            tail = pr[pr <= var]
            return -tail.mean() if len(tail) > 0 else -var

        return self._minimize(cvar_obj)

    # ------------------------------------------------------------------
    # 9. Maximo Sortino
    # ------------------------------------------------------------------
    def max_sortino(self) -> np.ndarray:
        R = self.returns.values
        rf_d = (1 + self.rf) ** (1 / self.periods) - 1

        def neg_sortino(w):
            pr = R @ w
            downside = pr[pr < rf_d] - rf_d
            dd = (
                max(np.sqrt(np.mean(downside**2)) * np.sqrt(self.periods), 1e-8)
                if len(downside) > 0
                else 1e-8
            )
            return -(pr.mean() * self.periods - self.rf) / dd

        return self._minimize(neg_sortino)

    # ------------------------------------------------------------------
    # 10. Maximo Calmar
    # ------------------------------------------------------------------
    def max_calmar(self) -> np.ndarray:
        R = self.returns.values

        def neg_calmar(w):
            pr = R @ w
            cum = np.cumprod(1 + pr)
            mdd = abs(((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)).min())
            mdd = max(mdd, 1e-8)
            return -(pr.mean() * self.periods) / mdd

        return self._minimize(neg_calmar)

    # ------------------------------------------------------------------
    # 11. Hierarchical Risk Parity (HRP)
    # ------------------------------------------------------------------
    def hrp(self) -> np.ndarray:
        return hierarchical_risk_parity(self.cov_matrix, self.corr_matrix, self.tickers)

    # ------------------------------------------------------------------
    # 12. Maxima Decorrelacion
    # ------------------------------------------------------------------
    def max_decorrelation(self) -> np.ndarray:
        corr = self.corr_matrix.values
        return self._minimize(lambda w: w @ corr @ w)

    # ------------------------------------------------------------------
    # 13. Minimo CDaR
    # ------------------------------------------------------------------
    def min_cdar(self, alpha: float = 0.05) -> np.ndarray:
        R = self.returns.values

        def cdar_obj(w):
            cum = np.cumprod(1 + R @ w)
            losses = -((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum))
            var_dd = np.quantile(losses, 1 - alpha)
            tail = losses[losses >= var_dd]
            return tail.mean() if len(tail) > 0 else var_dd

        return self._minimize(cdar_obj)

    # ------------------------------------------------------------------
    # 14. Volatilidad Objetivo (10%)
    # ------------------------------------------------------------------
    def target_volatility(self, target_vol: float = 0.10) -> np.ndarray:
        mu, cov = self.mean_returns.values, self.cov_matrix.values
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "ineq", "fun": lambda w: target_vol - np.sqrt(max(w @ cov @ w, 0))},
        )
        w = self._minimize(lambda w: -(w @ mu), constraints=constraints)
        if np.sqrt(w @ cov @ w) > target_vol * 1.5:
            return self.min_variance()
        return w

    # ------------------------------------------------------------------
    # 15. Maximo Retorno (Long-only, sin restriccion de riesgo)
    # ------------------------------------------------------------------
    def max_return(self) -> np.ndarray:
        mu = self.mean_returns.values

        # Concentracion pura en el activo de mayor retorno esperado
        # suavizada con regularizacion L2 para evitar solucion degenerada
        def neg_ret_reg(w):
            return -(w @ mu) + 0.01 * np.sum(w**2)

        return self._minimize(neg_ret_reg)

    # ------------------------------------------------------------------
    # 16. Minimo Skewness Negativo (maximiza la asimetria positiva)
    # ------------------------------------------------------------------
    def min_neg_skewness(self) -> np.ndarray:
        R = self.returns.values

        def neg_skew_obj(w):
            pr = R @ w
            std = pr.std()
            if std < 1e-10:
                return 0.0
            return -float(((pr - pr.mean()) ** 3).mean() / std**3)

        return self._minimize(neg_skew_obj)

    # ------------------------------------------------------------------
    # 17. Kelly Fraction (fraccion de Kelly optima)
    # ------------------------------------------------------------------
    def kelly_fraction(self) -> np.ndarray:
        """Maximiza el logaritmo del retorno esperado (criterio de Kelly)."""
        R = self.returns.values

        def neg_log_growth(w):
            pr = R @ w
            if np.any(pr <= -1):
                return 1e8
            return -np.mean(np.log1p(pr))

        return self._minimize(neg_log_growth)

    # ------------------------------------------------------------------
    # 18. Black-Litterman (prior de mercado, sin views activos)
    # ------------------------------------------------------------------
    def black_litterman(self) -> np.ndarray:
        """
        Black-Litterman sin views: usa el retorno de equilibrio implicito
        del mercado (reverse optimization sobre el portafolio de mercado
        aproximado como equal weight) para calcular los pesos optimos.
        """
        cov = self.cov_matrix.values
        # Pesos de mercado aproximados (equal weight como proxy)
        w_mkt = np.repeat(1.0 / self.n, self.n)
        # Lambda de aversion al riesgo calibrado del mercado
        mkt_ret = float(w_mkt @ self.mean_returns.values)
        mkt_vol = float(np.sqrt(w_mkt @ cov @ w_mkt))
        lam = (mkt_ret - self.rf) / (mkt_vol**2) if mkt_vol > 0 else 3.0
        lam = np.clip(lam, 0.5, 10.0)
        # Retornos de equilibrio (Pi)
        pi = lam * cov @ w_mkt
        # Sin views: mu_BL = pi
        mu_bl = pi

        # Optimizacion con mu_BL
        def neg_sharpe_bl(w):
            ret = w @ mu_bl
            vol = np.sqrt(max(w @ cov @ w, 1e-12))
            return -(ret - self.rf) / vol

        return self._minimize(neg_sharpe_bl)

    # ------------------------------------------------------------------
    # 19. Minima Kurtosis de Cola (minimiza kurtosis del portafolio)
    # ------------------------------------------------------------------
    def min_tail_kurtosis(self) -> np.ndarray:
        R = self.returns.values

        def kurtosis_obj(w):
            pr = R @ w
            std = pr.std()
            if std < 1e-10:
                return 0.0
            return float(((pr - pr.mean()) ** 4).mean() / std**4)

        return self._minimize(kurtosis_obj)

    # ------------------------------------------------------------------
    # 20. Maximo Omega Ratio (umbral = tasa libre de riesgo diaria)
    # ------------------------------------------------------------------
    def max_omega(self, threshold: float | None = None) -> np.ndarray:
        """
        El Omega Ratio mide la relacion entre ganancias y perdidas respecto
        a un umbral, usando toda la distribucion de retornos (no solo media
        y varianza). Maximizarlo favorece portafolios con mas probabilidad
        de superar el umbral y perdidas mas acotadas cuando no lo logran.
        """
        R = self.returns.values
        thr = threshold if threshold is not None else (1 + self.rf) ** (1 / self.periods) - 1

        def neg_omega(w):
            pr = R @ w
            excess = pr - thr
            gains = excess[excess > 0].sum()
            losses = -excess[excess < 0].sum()
            if losses < 1e-10:
                return -1e6 if gains > 0 else 0.0
            return -(gains / losses)

        return self._minimize(neg_omega)

    # ------------------------------------------------------------------
    # 21. Maxima Entropia de Pesos (diversificacion por entropia de Shannon)
    # ------------------------------------------------------------------
    def max_entropy(self) -> np.ndarray:
        """
        Maximiza la entropia de Shannon de los pesos, penalizada por la
        varianza del portafolio. Produce carteras muy diversificadas
        (los pesos se acercan entre si) sin ignorar el riesgo.
        """
        cov = self.cov_matrix.values
        eps = 1e-10

        def neg_entropy_obj(w):
            w_pos = np.clip(w, eps, None)
            w_pos = w_pos / w_pos.sum()
            entropy = -np.sum(w_pos * np.log(w_pos))
            port_var = w @ cov @ w
            # Penalizacion suave por riesgo, recompensa por entropia
            return -entropy + 2.0 * port_var

        return self._minimize(neg_entropy_obj)

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------
    def optimize(self, strategy: str, **kwargs) -> np.ndarray:
        strategies = self.available_strategies()
        if strategy not in strategies:
            raise ValueError(f"Estrategia desconocida: {strategy}")
        return strategies[strategy](**kwargs) if kwargs else strategies[strategy]()

    def available_strategies(self) -> dict:
        return {
            "Equal Weight (1/N)": self.equal_weight,
            "Maximo Sharpe": self.max_sharpe,
            "Minima Varianza": self.min_variance,
            "Paridad de Riesgo (ERC)": self.risk_parity,
            "Inverso a Volatilidad": self.inverse_volatility,
            "Inverso a Varianza": self.inverse_variance,
            "Maxima Diversificacion": self.max_diversification,
            "Minimo CVaR": self.min_cvar,
            "Maximo Sortino": self.max_sortino,
            "Maximo Calmar": self.max_calmar,
            "Hierarchical Risk Parity (HRP)": self.hrp,
            "Maxima Decorrelacion": self.max_decorrelation,
            "Minimo CDaR": self.min_cdar,
            "Volatilidad Objetivo (10%)": self.target_volatility,
            "Maximo Retorno": self.max_return,
            "Minimo Skewness Negativo": self.min_neg_skewness,
            "Kelly Fraction": self.kelly_fraction,
            "Black-Litterman (equilibrio)": self.black_litterman,
            "Minima Kurtosis de Cola": self.min_tail_kurtosis,
            "Maximo Omega Ratio": self.max_omega,
            "Maxima Entropia": self.max_entropy,
        }

    def weights_to_series(self, w: np.ndarray) -> pd.Series:
        return pd.Series(w, index=self.tickers)


STRATEGY_NAMES = [
    "Equal Weight (1/N)",
    "Maximo Sharpe",
    "Minima Varianza",
    "Paridad de Riesgo (ERC)",
    "Inverso a Volatilidad",
    "Inverso a Varianza",
    "Maxima Diversificacion",
    "Minimo CVaR",
    "Maximo Sortino",
    "Maximo Calmar",
    "Hierarchical Risk Parity (HRP)",
    "Maxima Decorrelacion",
    "Minimo CDaR",
    "Volatilidad Objetivo (10%)",
    "Maximo Retorno",
    "Minimo Skewness Negativo",
    "Kelly Fraction",
    "Black-Litterman (equilibrio)",
    "Minima Kurtosis de Cola",
    "Maximo Omega Ratio",
    "Maxima Entropia",
]
