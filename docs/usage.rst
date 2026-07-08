Usage
=====

Import the package as a flat function namespace:

.. code-block:: python

   import AbaQuant as aq

.. tab-set::

    .. tab-item:: Financial Math

       .. code-block:: python

          # Present/future value and annuities
          aq.valor_presente(1100, 0.10, 1)
          aq.vp_anualidad_efectiva(100, 0.01, 12)

          # Bond pricing and interest-rate risk
          aq.precio_bono(1000, 0.08, 40, 0.09, 10)
          aq.riesgo_bono(1000, 0.08, 40, 0.09, 10, 2)  # duration & convexity

          # Corporate valuation
          aq.capm_cost_of_equity(0.04, 1.2, 0.10)
          aq.dcf_valuation(100, 0.10, 0.03, 0.08, 5, 500, 100)

          # Market risk
          aq.calcular_var_parametrico(0.10, 0.20, 1_000_000, 0.95, 10)

    .. tab-item:: Derivatives

       .. code-block:: python

          # Forwards
          aq.forward_price(100, 0.05, 1, carry=0.02)
          aq.fra(0.04, 0.05, 1, 2, 1_000_000, 0.055)

          # Vanilla options + Greeks
          aq.black_scholes(100, 100, 0.05, 0.20, 1)
          aq.calcular_griegas(100, 100, 0.05, 0.20, 1, es_call=True)

          # Binomial tree (American/European)
          aq.arbol_binomial_crr(100, 100, 0.05, 0.20, 1, 200, True, True)

          # Exotics
          aq.barrera_down_and_out(100, 95, 80, 1, 0.05, 0.20)
          aq.opciones_asiaticas_aritmeticas(100, 100, 1, 0.05, 0.20)
          aq.opciones_lookback_flotante(100, 90, 1, 0.05, 0.20)

    .. tab-item:: Advanced Models

       .. code-block:: python

          from AbaQuant.derivados_avanzados import (
              BSMEngine, HestonEngine, MertonEngine, SABREngine,
              calibrate_heston, compare_all_models,
          )

          heston = HestonEngine(S=100, K=100, T=1, r=0.05, q=0.0,
                                 v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
          heston.call_price()

          calibrate_heston(S=100, T=1, r=0.05, q=0.0,
                            strikes=[90, 100, 110], market_ivs=[0.24, 0.20, 0.22])

          compare_all_models(S=100, K=100, T=1, r=0.05, sigma=0.20)

    .. tab-item:: Credit Risk

       .. code-block:: python

          tm = aq.build_transition_matrix()
          values = aq.bond_values_per_rating(1000, 0.05, 3, 1, 0.4, aq.DEFAULT_SPREADS)

          # CDS fair spread
          aq.valuar_cds(hazard_rate=0.02, discount_rate=0.05, maturity=5, recovery_rate=0.4)

          # Correlated portfolio default simulation
          aq.gaussian_copula_simulation(bonds_data, tm, corr_mat, n_sims=50_000)

    .. tab-item:: Portfolio Optimization

       .. code-block:: python

          from AbaQuant.portafolioopt import (
              PortfolioOptimizer, markowitz_frontier, run_backtest, compute_all_metrics,
          )

          optimizer = PortfolioOptimizer(returns_df, rf=0.04)
          weights = optimizer.optimize("Maximo Sharpe")

          metrics = compute_all_metrics(returns_df @ weights, rf=0.04)

          backtest = run_backtest(prices_df, "Maximo Sharpe", rebalance_freq="Mensual")