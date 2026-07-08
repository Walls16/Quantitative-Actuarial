Development
===========

.. image:: https://img.shields.io/badge/env-conda%20%22quact%22-blue
   :alt: conda environment

Environment
-----------

The project is currently developed and verified in the ``AQ`` conda
environment:

.. code-block:: powershell

   conda activate quact
   python -m pip install -e ".[dev,docs]"

Tests
-----

The test suite stores deterministic expected outputs in
``tests/fixtures/results.json`` and compares every public function against
those saved results.

.. code-block:: powershell

   conda run -n quact pytest -q

Documentation
-------------

Build the Sphinx site locally:

.. code-block:: powershell

   conda run -n quact sphinx-build -b html docs docs/_build/html

The generated HTML appears in ``docs/_build/html``.

GitHub Pages
------------

The repository includes a GitHub Actions workflow at
``.github/workflows/docs.yml``. On pushes to ``main``, it builds the Sphinx
site and deploys the HTML artifact to GitHub Pages.

In the GitHub repository settings, set Pages to deploy from GitHub Actions.