from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

import quantitativeactuarial as quact


def sample_bonds() -> list[dict[str, Any]]:
    return [
        {
            "nombre": "A",
            "rating_idx": 0,
            "values": np.linspace(100.0, 80.0, 18),
        },
        {
            "nombre": "B",
            "rating_idx": 8,
            "values": np.linspace(90.0, 50.0, 18),
        },
    ]


def sample_distribution() -> list[tuple[float, float]]:
    return quact.independent_distribution(sample_bonds(), quact.DEFAULT_TM)


def sample_simulations() -> np.ndarray:
    return quact.gaussian_copula_simulation(
        [{"rating_idx": 0, "values": np.arange(18, dtype=float)}],
        quact.DEFAULT_TM,
        np.eye(1),
        n_sims=5,
        seed=11,
    )


CASES: dict[str, Callable[[], Any]] = {
    "arbol_binomial_crr": lambda: quact.arbol_binomial_crr(100, 100, 0.05, 0.2, 1, 3, True, False, 0),
    "barrera_down_and_out": lambda: quact.barrera_down_and_out(100, 95, 80, 1, 0.05, 0.2, 0.01, "call"),
    "binomial_tree": lambda: quact.binomial_tree(100, 100, 1, 0.05, 0.2, 3, 0, "call", False),
    "black_76": lambda: quact.black_76(100, 100, 0.05, 0.2, 1, True),
    "black_scholes": lambda: quact.black_scholes(100, 100, 0.05, 0.2, 1, True, 0),
    "bond_values_per_rating": lambda: quact.bond_values_per_rating(1000, 0.05, 3, 1, 0.4, quact.DEFAULT_SPREADS),
    "build_transition_matrix": lambda: quact.build_transition_matrix(nr_treatment="redistribute"),
    "calcular_griegas": lambda: quact.calcular_griegas(100, 100, 0.05, 0.2, 1, True, 0),
    "calcular_payoff_leg": lambda: quact.calcular_payoff_leg("call", 1, np.array([90, 100, 110]), 100, 2),
    "calcular_var_cvar_montecarlo": lambda: quact.calcular_var_cvar_montecarlo(
        0.1, 0.2, 1000, 0.95, 10, simulaciones=1000, seed=7
    ),
    "calcular_var_parametrico": lambda: quact.calcular_var_parametrico(0.1, 0.2, 1000, 0.95, 10),
    "calcular_vp_dividendos": lambda: quact.calcular_vp_dividendos(10, 4, 0.08, 1, "Continua"),
    "calcular_vp_flujos_irregulares": lambda: quact.calcular_vp_flujos_irregulares([100, 200, 300], [0.5, 1, 2], 0.08, "Continua"),
    "desglosar_periodos": lambda: quact.desglosar_periodos(2.345),
    "expected_value_and_sigma": lambda: quact.expected_value_and_sigma(sample_bonds(), quact.DEFAULT_TM),
    "forward_calculo": lambda: quact.forward_calculo(100, 0.05, 0.02, 1, "Continua"),
    "forward_price": lambda: quact.forward_price(100, 0.05, 1, 0.02),
    "fra": lambda: quact.fra(0.04, 0.05, 1, 2, 1_000_000, 0.055),
    "gaussian_copula_simulation": sample_simulations,
    "generar_tabla_reinversion": lambda: quact.generar_tabla_reinversion(1000, 0.12, 2),
    "griegas_bsm": lambda: quact.griegas_bsm("Yield", 100, 100, 1, 0.05, 0.2, 0.01),
    "independent_distribution": sample_distribution,
    "nper_anualidad_vf": lambda: quact.nper_anualidad_vf(1268.2503013196977, 100, 0.01),
    "nper_anualidad_vp": lambda: quact.nper_anualidad_vp(1125.5077473484641, 100, 0.01),
    "nper_gradiente_arit_vf": lambda: quact.nper_gradiente_arit_vf(660, 100, 5, 0.01),
    "nper_gradiente_arit_vp": lambda: quact.nper_gradiente_arit_vp(560, 100, 5, 0.01),
    "nper_gradiente_geo_vf": lambda: quact.nper_gradiente_geo_vf(700, 100, 0.02, 0.01),
    "nper_gradiente_geo_vp": lambda: quact.nper_gradiente_geo_vp(560, 100, 0.02, 0.01),
    "numero_periodos": lambda: quact.numero_periodos(1000, 1469.3280768, 0.08),
    "opcion_chooser_simple": lambda: quact.opcion_chooser_simple(100, 100, 0.5, 1, 0.05, 0.2, 0.01),
    "opcion_perpetua": lambda: quact.opcion_perpetua(120, 100, 0.05, 0.2, True),
    "opciones_asiaticas_aritmeticas": lambda: quact.opciones_asiaticas_aritmeticas(100, 100, 1, 0.05, 0.2, 0, "call"),
    "opciones_asiaticas_geometricas": lambda: quact.opciones_asiaticas_geometricas(100, 100, 1, 0.05, 0.2, 0, "call"),
    "opciones_asset_or_nothing": lambda: quact.opciones_asset_or_nothing(100, 100, 1, 0.05, 0.2, 0, "call"),
    "opciones_bsm": lambda: quact.opciones_bsm("Yield", 100, 100, 1, 0.05, 0.2, 0.01),
    "opciones_cash_or_nothing": lambda: quact.opciones_cash_or_nothing(100, 100, 10, 1, 0.05, 0.2, 0, "call"),
    "opciones_compuestas": lambda: quact.opciones_compuestas(100, 5, 100, 0.5, 1, 0.05, 0.2, 0.01, "call_on_call"),
    "opciones_gap": lambda: quact.opciones_gap(100, 95, 100, 1, 0.05, 0.2, 0.01, "call"),
    "opciones_intercambio_uxv": lambda: quact.opciones_intercambio_uxv(95, 100, 0.01, 0.02, 0.2, 0.25, 0.3, 1),
    "opciones_lookback_flotante": lambda: quact.opciones_lookback_flotante(100, 90, 1, 0.05, 0.2, 0.01, "call"),
    "perfil_estrategia": lambda: quact.perfil_estrategia(
        100,
        [{"tipo": "call", "posicion": 1, "K": 100, "prima": 2}],
        puntos=5,
    ),
    "precio_bono": lambda: quact.precio_bono(1000, 0.08, 40, 0.09, 10),
    "precio_forward": lambda: quact.precio_forward(100, 0.05, 1),
    "precio_forward_commodity": lambda: quact.precio_forward_commodity(100, 0.05, 0.01, 1),
    "precio_forward_dividendo_continuo": lambda: quact.precio_forward_dividendo_continuo(100, 0.05, 0.02, 1),
    "precio_forward_dividendos_discretos": lambda: quact.precio_forward_dividendos_discretos(100, 0.05, 1, 3),
    "precio_forward_divisa": lambda: quact.precio_forward_divisa(20, 0.08, 0.04, 1),
    "rendimiento_requerido_accion": lambda: quact.rendimiento_requerido_accion(5, 100, 0.03),
    "riesgo_bono": lambda: quact.riesgo_bono(1000, 0.08, 40, 0.09, 10, 2),
    "scale_var_cvar": lambda: quact.scale_var_cvar(quact.var_cvar_parametric(1000, 25, (0.95,)), (0.95,)),
    "tabla_amortizacion": lambda: quact.tabla_amortizacion(1000, 0.01, 3),
    "tasa_efectiva_a_nominal": lambda: quact.tasa_efectiva_a_nominal(0.12682503013196977, 12),
    "tasa_instantanea_a_efectiva": lambda: quact.tasa_instantanea_a_efectiva(0.12),
    "tasa_instantanea_a_nominal": lambda: quact.tasa_instantanea_a_nominal(0.12, 12),
    "tasa_nominal_a_efectiva": lambda: quact.tasa_nominal_a_efectiva(0.12, 12),
    "tasa_nominal_a_instantanea": lambda: quact.tasa_nominal_a_instantanea(0.12, 12),
    "tasa_nominal_m_a_nominal_p": lambda: quact.tasa_nominal_m_a_nominal_p(0.12, 12, 4),
    "tasa_rendimiento": lambda: quact.tasa_rendimiento(1000, 1469.3280768, 5),
    "tasa_rendimiento_bono": lambda: quact.tasa_rendimiento_bono(950, 1000, 0.08, 40, 10),
    "thresholds_per_bond": lambda: quact.thresholds_per_bond(0, quact.DEFAULT_TM),
    "valor_forward_calculo": lambda: quact.valor_forward_calculo(100, 102, 0.05, 0.02, 0.5, "Larga", "Continua"),
    "valor_forward_en_vida": lambda: quact.valor_forward_en_vida(100, 102, 0.05, 0.02, 0.5),
    "valor_futuro": lambda: quact.valor_futuro(1000, 0.08, 5),
    "valor_futuro_continuo": lambda: quact.valor_futuro_continuo(1000, 0.08, 5),
    "valor_presente": lambda: quact.valor_presente(1469.3280768, 0.08, 5),
    "valor_presente_continuo": lambda: quact.valor_presente_continuo(1500, 0.08, 5),
    "valuacion_gordon_shapiro": lambda: quact.valuacion_gordon_shapiro(5, 0.12, 0.03),
    "valuacion_multiplos": lambda: quact.valuacion_multiplos(12, 8),
    "var_cvar_from_distribution": lambda: quact.var_cvar_from_distribution(sample_distribution(), (0.95,)),
    "var_cvar_from_simulations": lambda: quact.var_cvar_from_simulations(sample_simulations(), (0.95,)),
    "var_cvar_parametric": lambda: quact.var_cvar_parametric(1000, 25, (0.95,)),
    "vf_anualidad_continua": lambda: quact.vf_anualidad_continua(1200, 0.08, 3),
    "vf_anualidad_efectiva": lambda: quact.vf_anualidad_efectiva(100, 0.01, 12),
    "vf_anualidad_nominal": lambda: quact.vf_anualidad_nominal(100, 0.12, 12, 12, 1),
    "vf_gradiente_aritmetico": lambda: quact.vf_gradiente_aritmetico(100, 5, 0.01, 6),
    "vf_gradiente_geo": lambda: quact.vf_gradiente_geo(100, 0.02, 0.01, 6),
    "vp_anualidad_continua": lambda: quact.vp_anualidad_continua(1200, 0.08, 3),
    "vp_anualidad_efectiva": lambda: quact.vp_anualidad_efectiva(100, 0.01, 12),
    "vp_anualidad_nominal": lambda: quact.vp_anualidad_nominal(100, 0.12, 12, 12, 1),
    "vp_gradiente_aritmetico": lambda: quact.vp_gradiente_aritmetico(100, 5, 0.01, 6),
    "vp_gradiente_geo": lambda: quact.vp_gradiente_geo(100, 0.02, 0.01, 6),
    "vp_perpetuidad": lambda: quact.vp_perpetuidad(100, 0.05),
}
