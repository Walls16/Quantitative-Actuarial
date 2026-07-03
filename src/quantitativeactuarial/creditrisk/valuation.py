"""CreditMetrics bond valuation by destination rating."""

from __future__ import annotations

import numpy as np


def bond_values_per_rating(VN: float, cupon_pct: float, T: int,
                            pagos_ano: int, recovery_pct: float,
                            spreads: np.ndarray,
                            include_d: bool = True,
                            spread_times: np.ndarray | None = None) -> np.ndarray:
    """
    Value a bond under each possible destination rating.

    Parameters
    ----------
    VN:
        Face value.
    cupon_pct:
        Annual coupon rate, e.g. ``0.05`` for 5%.
    T:
        Years to maturity.
    pagos_ano:
        Coupon payments per year.
    recovery_pct:
        Recovery rate in default, e.g. ``0.43`` for 43%.
    spreads:
        All-in yield table by rating and tenor.
    include_d:
        If ``True``, include the default state valuation.
    spread_times:
        Optional tenor times matching the spread columns. If omitted, annual
        tenors ``[1, 2, ...]`` are assumed.

    Returns
    -------
    numpy.ndarray
        Values by rating destination.
    """
    n_rated = 17
    n_cols  = spreads.shape[1]

    # Construct time grid for the yield curve columns
    if spread_times is None:
        times = np.arange(1, n_cols + 1, dtype=float)   # [1,2,…,n_cols]
    else:
        times = np.asarray(spread_times, dtype=float)

    vals  = np.zeros(n_rated + (1 if include_d else 0))
    cupon = VN * cupon_pct / pagos_ano
    # Cash flow times
    cf_times = np.array([(t + 1) / pagos_ano for t in range(T * pagos_ano)])

    for r_idx in range(n_rated):
        y_row = spreads[r_idx]
        pv = 0.0
        for t_yr in cf_times:
            # Find the yield for this cash-flow time via nearest available tenor
            idx = int(np.argmin(np.abs(times - t_yr)))
            # If the exact time is beyond the last available, use the last
            if t_yr > times[-1]:
                idx = len(times) - 1
            y = y_row[idx]
            pv += cupon / (1.0 + y) ** t_yr
        # Principal discount at maturity T
        idx_T = int(np.argmin(np.abs(times - T)))
        if T > times[-1]:
            idx_T = len(times) - 1
        pv += VN / (1.0 + y_row[idx_T]) ** T
        vals[r_idx] = pv

    if include_d:
        vals[17] = recovery_pct * VN

    return vals

__all__ = ["bond_values_per_rating"]
