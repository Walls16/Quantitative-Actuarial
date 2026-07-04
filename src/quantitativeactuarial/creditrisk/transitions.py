"""Credit-rating transition matrix construction."""

from __future__ import annotations

import numpy as np

from .data import RATINGS_EMIT, _TM_RAW_17x19


def build_transition_matrix(
    raw_17x19: np.ndarray | None = None, nr_treatment: str = "redistribute"
) -> np.ndarray:
    """
    Construye la matriz de transición de trabajo a partir de la raw 17x19.

    nr_treatment:
      'redistribute'    → redistribuye NR proporcionalmente entre AAA..D
                          → retorna (18×18), filas suman 1 exactamente
      'simple_normalize'→ descarta NR, renormaliza los 18 cols (AAA..D) a 1
                          → retorna (18×18), filas suman 1 exactamente
      'raw_no_d_nr'     → usa AAA..CCC/C tal como aparecen en la fuente,
                          SIN normalizar, SIN columna D.
                          Las filas NO suman 1 (ejercicio de clase clásico).
                          → retorna (17×17) para que la lógica sea coherente.
                          En este modo bond_values solo calcula 17 valores.
    """
    if raw_17x19 is None:
        raw_17x19 = _TM_RAW_17x19.copy()

    raw = raw_17x19.astype(float).copy()  # (17, 19)

    # Separar columnas: 0..16=AAA..CCC/C, 17=D, 18=NR
    rated_probs = raw[:, :17]  # AAA..CCC/C  (17 cols)
    d_probs = raw[:, 17]  # D
    nr_probs = raw[:, 18]  # NR

    if nr_treatment == "raw_with_d":
        # Probabilidades S&P crudas, columnas AAA..D (18 destinos), SIN NR.
        # Las filas NO suman 1 — NR se excluye sin redistribuir (= método Excel).
        # Este es el modo clásico de los libros de texto / ejercicios de CreditMetrics.
        # Devuelve (18, 18): filas 0..16 suman < 1;  fila D = estado absorbente.
        raw18 = np.column_stack([rated_probs, d_probs])  # (17, 18)
        d_row = np.zeros(18)
        d_row[-1] = 1.0
        return np.vstack([raw18, d_row])  # (18, 18)

    elif nr_treatment == "redistribute":
        # Redistribuir NR proporcionalmente entre los 18 estados (AAA..CCC/C + D)
        rated_plus_d = np.column_stack([rated_probs, d_probs])  # (17, 18)
        sums_rated_d = rated_plus_d.sum(axis=1, keepdims=True)
        sums_rated_d[sums_rated_d == 0] = 1.0
        adjusted = rated_plus_d + nr_probs[:, None] * (rated_plus_d / sums_rated_d)
        # Normalize to sum exactly 1
        rs = adjusted.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        adjusted = adjusted / rs  # (17, 18)
        d_row = np.zeros(18)
        d_row[-1] = 1.0
        return np.vstack([adjusted, d_row])  # (18, 18)

    elif nr_treatment == "simple_normalize":
        # Descartar NR, renormalizar (AAA..D) a 1 via escala simple
        rated_plus_d = np.column_stack([rated_probs, d_probs])  # (17, 18)
        sums = rated_plus_d.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        adjusted = rated_plus_d / sums  # (17, 18)
        d_row = np.zeros(18)
        d_row[-1] = 1.0
        return np.vstack([adjusted, d_row])  # (18, 18)

    else:  # 'raw_no_d_nr'
        # Usar AAA..CCC/C tal como son, sin D, sin NR, sin normalizar
        # Retorna (17, 17) — coherente para ejercicios de clase
        # NOTA: las filas suman < 1 porque NR y D están excluidos
        # En este modo se asume que el residuo "desaparece" (no default, no NR)
        return rated_probs.copy()  # (17, 17)


DEFAULT_TM = build_transition_matrix()
_TM_SIZE_BY_MODE = {"raw_with_d": 18, "redistribute": 18, "simple_normalize": 18, "raw_no_d_nr": 17}
RATINGS_NO_D = RATINGS_EMIT[:17]

__all__ = [
    "build_transition_matrix",
    "DEFAULT_TM",
    "_TM_SIZE_BY_MODE",
    "RATINGS_NO_D",
]
