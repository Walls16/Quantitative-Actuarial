Usage
=====

Import the package as a function namespace:

.. code-block:: python

   import quantitativeactuarial as quact

Financial Math
--------------

.. code-block:: python

   quact.valor_presente(1100, 0.10, 1)
   quact.vp_anualidad_efectiva(100, 0.01, 12)
   quact.precio_bono(1000, 0.08, 40, 0.09, 10)

Derivatives
-----------

.. code-block:: python

   quact.forward_price(100, 0.05, 1, carry=0.02)
   quact.black_scholes(100, 100, 0.05, 0.2, 1)
   quact.arbol_binomial_crr(100, 100, 0.05, 0.2, 1, 100)

Credit Risk
-----------

.. code-block:: python

   tm = quact.build_transition_matrix()
   values = quact.bond_values_per_rating(1000, 0.05, 3, 1, 0.4, quact.DEFAULT_SPREADS)
   thresholds = quact.thresholds_per_bond(0, tm)


ola owowow
----------

esta buena la pag, no?