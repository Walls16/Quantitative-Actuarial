"""Loan amortization schedules."""

from __future__ import annotations

import pandas as pd


def tabla_amortizacion(VP: float, i_m: float, n_m: int) -> pd.DataFrame:
    """Genera una tabla de amortización de pagos fijos sin el periodo 0."""
    # MEJORA: validar inputs y redondear n_m para evitar pérdida silenciosa del último periodo
    if i_m < 0:
        raise ValueError("i_m debe ser >= 0")
    n_m = round(n_m)  # evita que n_m=11.9 trunce a 11 y deje saldo residual

    if i_m == 0:
        pago = VP / n_m
    else:
        pago = VP * (i_m / (1 - (1 + i_m) ** (-n_m)))

    saldo = VP
    datos = []

    for t in range(1, n_m + 1):
        saldo_inicial = saldo
        interes = saldo_inicial * i_m
        amort = pago - interes

        # MEJORA: en el último periodo, liquidar exactamente para eliminar residuos de redondeo
        if t == n_m:
            amort = saldo_inicial
            saldo = 0.0
        else:
            saldo -= amort
            if abs(saldo) < 0.01:
                saldo = 0.0

        datos.append(
            {
                "Periodo": t,
                "Saldo Inicial": saldo_inicial,
                "Interés": interes,
                "Amortización": amort,
                "Saldo Insoluto": saldo,
            }
        )

    return pd.DataFrame(datos)


__all__ = ["tabla_amortizacion"]
