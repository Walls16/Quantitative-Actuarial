Credit Risk
============

Rating transition matrices, CDS and CDO valuation, Gaussian-copula
portfolio simulation, and credit VaR/CVaR — implementing the
CreditMetrics and Hull one-factor Gaussian copula methodologies.

Highlights
----------

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Transition matrix

      .. code-block:: python

         tm = aq.build_transition_matrix()

   .. grid-item-card:: Bond migration valuation

      .. code-block:: python

         aq.bond_values_per_rating(
             1000, 0.05, 3, 1, 0.4, aq.DEFAULT_SPREADS,
         )

   .. grid-item-card:: CDS fair spread

      .. code-block:: python

         aq.valuar_cds(
             hazard_rate=0.02, discount_rate=0.05,
             maturity=5, recovery_rate=0.4,
         )

   .. grid-item-card:: CDO tranche spread (Hull, one-factor)

      .. code-block:: python

         aq.valuar_tranche(
             h=0.02, rho=0.30, n=125, R=0.40,
             alpha_L=0.03, alpha_H=0.06,
             tasa_rf=0.03, periodos=[1, 2, 3],
             F_nodes=nodes, w_nodes=weights,
         )

   .. grid-item-card:: Correlated portfolio simulation

      .. code-block:: python

         aq.gaussian_copula_simulation(
             bonds_data, transition_matrix, corr_matrix, n_sims=50_000,
         )

Module Reference
----------------

.. automodule:: AbaQuant.creditrisk
   :members:
   :undoc-members:

Transition Matrices
--------------------

.. automodule:: AbaQuant.creditrisk.transitions
   :members:
   :undoc-members:

CDS
---

.. automodule:: AbaQuant.creditrisk.cds
   :members:
   :undoc-members:

CDO
---

.. automodule:: AbaQuant.creditrisk.cdo
   :members:
   :undoc-members:

Bond Migration Valuation
--------------------------

.. automodule:: AbaQuant.creditrisk.valuation
   :members:
   :undoc-members:

Portfolio Distribution
------------------------

.. automodule:: AbaQuant.creditrisk.distribution
   :members:
   :undoc-members:

Gaussian Copula
----------------

.. automodule:: AbaQuant.creditrisk.copula
   :members:
   :undoc-members:

Risk Metrics
------------

.. automodule:: AbaQuant.creditrisk.risk
   :members:
   :undoc-members:

Reference Data
--------------

.. automodule:: AbaQuant.creditrisk.data
   :members:
   :undoc-members:

Types
-----

.. automodule:: AbaQuant.creditrisk.types
   :members:
   :undoc-members: