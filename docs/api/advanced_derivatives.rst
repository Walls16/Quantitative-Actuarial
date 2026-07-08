Advanced Derivatives
=====================

Stochastic-volatility, jump-diffusion, and Lévy pricing engines migrated
from the VQD application: Heston, Merton, SABR, Variance Gamma, NIG,
Bachelier, calibration workflows, and Monte Carlo pricing.

Highlights
----------

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Heston (stochastic volatility)

      .. code-block:: python

         from AbaQuant.derivados_avanzados import HestonEngine

         h = HestonEngine(
             S=100, K=100, T=1, r=0.05, q=0.0,
             v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
         )
         h.call_price()

   .. grid-item-card:: Merton (jump-diffusion)

      .. code-block:: python

         from AbaQuant.derivados_avanzados import MertonEngine

         m = MertonEngine(
             S=100, K=100, T=1, r=0.05, sigma=0.20,
             lam=1.0, mu_j=-0.10, sigma_j=0.15,
         )
         m.call_price()

   .. grid-item-card:: SABR calibration

      .. code-block:: python

         from AbaQuant.derivados_avanzados import calibrate_sabr

         calibrate_sabr(
             forward=100, maturity=1,
             strikes=[90, 100, 110], market_ivs=[0.24, 0.20, 0.22],
             beta=0.5,
         )

   .. grid-item-card:: Model comparison

      .. code-block:: python

         from AbaQuant.derivados_avanzados import compare_all_models

         compare_all_models(S=100, K=100, T=1, r=0.05, sigma=0.20)
         # -> {"BSM": ..., "CRR": ..., "Heston": ..., "Merton": ...}

Public API
----------

.. automodule:: AbaQuant.derivados_avanzados
   :members:
   :undoc-members:

Models
------

.. automodule:: AbaQuant.derivados_avanzados.models
   :members:
   :undoc-members:
   :no-index:

Numerics
--------

.. automodule:: AbaQuant.derivados_avanzados.numerics
   :members:
   :undoc-members:
   :no-index:

Calibration
-----------

.. automodule:: AbaQuant.derivados_avanzados.calibration
   :members:
   :undoc-members:
   :no-index:

Simulation
----------

.. automodule:: AbaQuant.derivados_avanzados.simulation
   :members:
   :undoc-members:
   :no-index:

Analytics
---------

.. automodule:: AbaQuant.derivados_avanzados.analytics
   :members:
   :undoc-members:
   :no-index: