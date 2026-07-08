Derivatives
============

.. currentmodule:: quantitativeactuarial

Forward contracts, Black-Scholes-Merton, binomial trees, Greeks, exotic
options, and option-strategy payoff tools.

Highlights
----------

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Black-Scholes-Merton

      .. code-block:: python

         aq.black_scholes(100, 100, 0.05, 0.20, 1)
         aq.calcular_griegas(100, 100, 0.05, 0.20, 1, es_call=True)

   .. grid-item-card:: Binomial tree (CRR)

      .. code-block:: python

         aq.arbol_binomial_crr(
             100, 100, 0.05, 0.20, 1, 200,
             es_call=True, american=True,
         )

   .. grid-item-card:: Exotic options

      .. code-block:: python

         aq.barrera_down_and_out(100, 95, 80, 1, 0.05, 0.20)
         aq.opciones_asiaticas_aritmeticas(100, 100, 1, 0.05, 0.20)
         aq.opciones_lookback_flotante(100, 90, 1, 0.05, 0.20)
         aq.opciones_intercambio_uxv(95, 100, 0.01, 0.02, 0.20, 0.25, 0.3, 1)

   .. grid-item-card:: Forwards & FRA

      .. code-block:: python

         aq.forward_price(100, 0.05, 1, carry=0.02)
         aq.fra(0.04, 0.05, 1, 2, 1_000_000, 0.055)

Module Reference
----------------

.. automodule:: AbaQuant.derivatives
   :members:
   :undoc-members:

Forwards
--------

.. automodule:: AbaQuant.derivatives.forwards
   :members:
   :undoc-members:

Vanilla Options
---------------

.. automodule:: AbaQuant.derivatives.vanilla
   :members:
   :undoc-members:

Trees
-----

.. automodule:: AbaQuant.derivatives.trees
   :members:
   :undoc-members:

Exotics
-------

.. automodule:: AbaQuant.derivatives.exotics
   :members:
   :undoc-members:

Strategies
----------

.. automodule:: AbaQuant.derivatives.strategies
   :members:
   :undoc-members: