quantitativeactuarial
=====================

Pure actuarial and quantitative-finance routines packaged as
``quantitativeactuarial`` and commonly imported as ``quact``.

The library is organized as small, stateless function modules:

* ``financial_math`` for rates, time value of money, annuities, loans, bonds,
  equity, cash-flow discounting, and market-risk helpers.
* ``derivatives`` for forwards, vanilla option pricing, trees, exotics, and
  payoff strategy utilities.
* ``creditrisk`` for transition matrices, bond migration valuation, portfolio
  distributions, Gaussian copula simulation, and credit VaR/CVaR.

Installation
------------

For local development from this repository:

.. code-block:: powershell

   conda activate quact
   python -m pip install -e ".[docs,dev]"

Quick Example
-------------

.. code-block:: python

   import quantitativeactuarial as quact

   effective = quact.tasa_nominal_a_efectiva(0.12, 12)
   call = quact.black_scholes(100, 100, 0.05, 0.2, 1.0)
   transition = quact.DEFAULT_TM

Build The Docs
--------------

.. code-block:: powershell

   conda run -n quact sphinx-build -W -b html docs docs/_build/html

.. toctree::
   :maxdepth: 2
   :caption: Contents

   usage
   api/index
   development
