from __future__ import annotations

from pathlib import Path

import pytest

import quantitativeactuarial as quact


def test_discrete_forward_prices_use_discrete_growth_factors() -> None:
    assert quact.precio_forward(100, 0.05, 2, capitalizacion="Discreta") == pytest.approx(100 * 1.05**2)
    assert quact.precio_forward_dividendo_continuo(100, 0.05, 0.02, 2, capitalizacion="Discreta") == pytest.approx(
        100 * 1.05**2 / 1.02**2
    )
    assert quact.precio_forward_dividendos_discretos(100, 0.05, 2, 3, capitalizacion="Discreta") == pytest.approx(
        97 * 1.05**2
    )
    assert quact.precio_forward_commodity(100, 0.05, 0.01, 2, capitalizacion="Discreta") == pytest.approx(
        100 * 1.05**2 * 1.01**2
    )
    assert quact.precio_forward_divisa(20, 0.08, 0.04, 2, capitalizacion="Discreta") == pytest.approx(
        20 * 1.08**2 / 1.04**2
    )


def test_discrete_forward_value_uses_discrete_discount_factors() -> None:
    value = quact.valor_forward_en_vida(105, 102, 0.05, 0.02, 0.5, capitalizacion="Discreta")
    expected = 105 / 1.02**0.5 - 102 / 1.05**0.5
    assert value == pytest.approx(expected)


def test_discrete_fra_uses_discrete_forward_rate_and_discounting() -> None:
    forward_rate, value = quact.fra(0.045, 0.05, 0.5, 1.0, 1_000_000, 0.055, capitalizacion="Discreta")
    expected_rate = ((1.05**1.0) / (1.045**0.5)) ** 2 - 1
    expected_value = 1_000_000 * (expected_rate - 0.055) * 0.5 / 1.05

    assert forward_rate == pytest.approx(expected_rate)
    assert value == pytest.approx(expected_value)


def test_forwards_page_does_not_convert_discrete_rates_with_log_shortcut() -> None:
    page = Path("app/pages/10_Forwards.py").read_text(encoding="utf-8")

    assert "np.log(1 +" not in page
    assert "Truco del motor" not in page
