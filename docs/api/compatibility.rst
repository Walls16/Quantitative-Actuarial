Compatibility Facades
=======================

These thin modules keep older import paths working while the actual
implementation lives in the split packages below. Prefer importing from
``AbaQuant`` directly, or from the specific subpackage
(``financial_math``, ``derivatives``, ``creditrisk``) in new code.

Rates & Financial Math (``tasas``)
------------------------------------

.. automodule:: AbaQuant.tasas

Derivatives (``derivados``)
------------------------------

.. automodule:: AbaQuant.derivados

Credit Risk (``credito``)
----------------------------

.. automodule:: AbaQuant.credito