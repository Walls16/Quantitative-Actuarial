AbaQuant
=======================

.. rst-class:: lead

   A pure Python library for actuarial mathematics, quantitative finance,
   derivatives pricing, credit risk, and portfolio optimization —
   commonly imported as ````.

.. note::

   Function names are currently in Spanish (the library's original
   language) and are being progressively translated to English.
   Behavior and signatures stay stable during the transition; see each
   module page for the current names.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: Financial Math
      :link: api/financial_math
      :link-type: doc

      Rates, time value of money, annuities, bonds, equity, DCF, CAPM, VaR/CVaR.

   .. grid-item-card:: Derivatives
      :link: api/derivatives
      :link-type: doc

      Forwards, Black-Scholes-Merton, binomial trees, Greeks, exotics, strategies.

   .. grid-item-card:: Advanced Derivatives
      :link: api/derivados_avanzados
      :link-type: doc

      Heston, Merton jump-diffusion, SABR, Variance Gamma, NIG, Bachelier, Monte Carlo.

   .. grid-item-card:: Credit Risk
      :link: api/creditrisk
      :link-type: doc

      Rating transition matrices, CDS, CDO tranches, Gaussian copulas, credit VaR.

   .. grid-item-card:: Portfolio Optimization
      :link: api/portafolioopt
      :link-type: doc

      21 allocation strategies, efficient frontier, backtesting, stress testing.

   .. grid-item-card:: Streamlit App
      :link: https://github.com/Walls16/Quantitative-Actuarial

      Interactive interface built on top of the pure library.

Installation
------------

.. code-block:: powershell

   conda activate AQ
   python -m pip install -e ".[docs,dev]"

Quick example
-------------

.. code-block:: python

   import AbaQuant as aq

   # Rates
   effective_rate = aq.tasa_nominal_a_efectiva(0.12, 12)

   # Derivatives
   call_price = aq.black_scholes(100, 100, 0.05, 0.20, 1.0)

   # Credit risk
   transition_matrix = aq.DEFAULT_TM

   # Portfolio optimization
   from AbaQuant.portafolioopt import PortfolioOptimizer

Build the docs
--------------

.. code-block:: powershell

   conda run -n quact sphinx-build -W -b html docs docs/_build/html

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   usage
   api/index
   development