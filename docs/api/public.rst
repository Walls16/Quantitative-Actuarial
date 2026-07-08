Public Namespace
==================

The top-level ``AbaQuant`` namespace re-exports every public
function from ``financial_math``, ``derivatives``, ``derivados_avanzados``,
``creditrisk``, and ``portafolioopt`` — so most users only ever need:

.. code-block:: python

   import AbaQuant as aq

Selected Public Functions
---------------------------

.. currentmodule:: AbaQuant

.. autosummary::
   :nosignatures:

   tasa_nominal_a_efectiva
   valor_presente
   vp_anualidad_efectiva
   precio_bono
   riesgo_bono
   dcf_valuation
   capm_cost_of_equity
   calcular_var_parametrico
   forward_price
   black_scholes
   calcular_griegas
   arbol_binomial_crr
   barrera_down_and_out
   opciones_asiaticas_aritmeticas
   opciones_lookback_flotante
   build_transition_matrix
   bond_values_per_rating
   valuar_cds
   valuar_tranche
   gaussian_copula_simulation
   PortfolioOptimizer
   markowitz_frontier
   run_backtest
   run_all_scenarios