Portfolio Optimization
========================

Pure portfolio-optimization routines: 21 allocation strategies, an
analytic efficient frontier, walk-forward backtesting, and historical
stress testing. Does not depend on PyPortfolioOpt.

Highlights
----------

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: 21 optimization strategies

      .. code-block:: python

         from AbaQuant.portafolioopt import PortfolioOptimizer

         opt = PortfolioOptimizer(returns_df, rf=0.04)
         weights = opt.optimize("Maximo Sharpe")
         # Also: "Minima Varianza", "Paridad de Riesgo (ERC)",
         # "Hierarchical Risk Parity (HRP)", "Black-Litterman (equilibrio)",
         # "Kelly Fraction", "Maximo Omega Ratio", ...

   .. grid-item-card:: Efficient frontier

      .. code-block:: python

         from AbaQuant.portafolioopt import markowitz_frontier

         markowitz_frontier(mean_returns, cov_matrix, n_points=50)

   .. grid-item-card:: Walk-forward backtesting

      .. code-block:: python

         from AbaQuant.portafolioopt import run_backtest

         run_backtest(
             prices_df, "Maximo Sharpe",
             rebalance_freq="Mensual", lookback_days=252,
         )

   .. grid-item-card:: Historical stress testing

      .. code-block:: python

         from AbaQuant.portafolioopt import run_all_scenarios

         run_all_scenarios(prices_df, weights_series)
         # 2008 crisis, COVID-19, dot-com bubble, 2022 rate hikes

Module Reference
----------------

.. automodule:: AbaQuant.portafolioopt
   :members:
   :undoc-members:

Optimization (21 Strategies)
------------------------------

.. automodule:: AbaQuant.portafolioopt.optimization
   :members:
   :undoc-members:

Risk Metrics
------------

.. automodule:: AbaQuant.portafolioopt.risk_metrics
   :members:
   :undoc-members:

Efficient Frontier
--------------------

.. automodule:: AbaQuant.portafolioopt.efficient_frontier
   :members:
   :undoc-members:

Hierarchical Risk Parity
---------------------------

.. automodule:: AbaQuant.portafolioopt.hierarchical
   :members:
   :undoc-members:

Backtesting
-----------

.. automodule:: AbaQuant.portafolioopt.backtesting
   :members:
   :undoc-members:

Stress Testing
--------------

.. automodule:: AbaQuant.portafolioopt.stress_testing
   :members:
   :undoc-members:

Solvers
-------

.. automodule:: AbaQuant.portafolioopt.solvers
   :members:
   :undoc-members:

Data Utilities
--------------

.. automodule:: AbaQuant.portafolioopt.data
   :members:
   :undoc-members: