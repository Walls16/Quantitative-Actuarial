Financial Math
===============

.. currentmodule:: AbaQuant

Core building blocks for interest rates, the time value of money, annuities,
loans, bonds, equity valuation, corporate finance, and market risk.

Highlights
----------

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Rate conversions

      .. code-block:: python

         aq.tasa_nominal_a_efectiva(0.12, 12)
         aq.tasa_nominal_a_instantanea(0.12, 12)

   .. grid-item-card:: Bond pricing & risk

      .. code-block:: python

         aq.precio_bono(1000, 0.08, 40, 0.09, 10)
         aq.riesgo_bono(1000, 0.08, 40, 0.09, 10, 2)

   .. grid-item-card:: DCF valuation

      .. code-block:: python

         aq.dcf_valuation(100, 0.10, 0.03, 0.08, 5, 500, 100)
         aq.capm_cost_of_equity(0.04, 1.2, 0.10)

   .. grid-item-card:: Market risk (VaR/CVaR)

      .. code-block:: python

         aq.calcular_var_parametrico(0.10, 0.20, 1_000_000, 0.95, 10)
         aq.calcular_var_cvar_montecarlo(0.10, 0.20, 1_000_000, 0.95, 10)

Module Reference
----------------

.. automodule:: AbaQuant.financial_math
   :members:
   :undoc-members:

Rates
-----

.. automodule:: AbaQuant.financial_math.rates
   :members:
   :undoc-members:

Time Value of Money
--------------------

.. automodule:: AbaQuant.financial_math.tvm
   :members:
   :undoc-members:

Annuities
---------

.. automodule:: AbaQuant.financial_math.annuities
   :members:
   :undoc-members:

Loans
-----

.. automodule:: AbaQuant.financial_math.loans
   :members:
   :undoc-members:

Bonds
-----

.. automodule:: AbaQuant.financial_math.bonds
   :members:
   :undoc-members:

Equity
------

.. automodule:: AbaQuant.financial_math.equity
   :members:
   :undoc-members:

Corporate Finance
-----------------

.. automodule:: AbaQuant.financial_math.corporate
   :members:
   :undoc-members:

Cash Flows
----------

.. automodule:: AbaQuant.financial_math.cashflows
   :members:
   :undoc-members:

Market Risk
-----------

.. automodule:: AbaQuant.financial_math.risk
   :members:
   :undoc-members:

Portfolio Primitives
---------------------

.. automodule:: AbaQuant.financial_math.portfolio
   :members:
   :undoc-members: