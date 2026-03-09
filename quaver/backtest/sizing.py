"""Risk-based position sizing utilities."""

from __future__ import annotations


def size_by_risk(
    account_value: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
    min_quantity: float = 1.0,
) -> float:
    """Compute a position size so that hitting *stop_loss* loses at most
    *risk_pct* of *account_value*.

    The formula is::

        risk_per_unit = |entry_price - stop_loss|
        quantity = (account_value * risk_pct) / risk_per_unit

    The result is floored to *min_quantity* (default ``1.0``) when the
    computed quantity is too small.

    :param account_value: Current account equity in dollars.
    :type account_value: float
    :param risk_pct: Maximum fraction of account to risk (e.g. ``0.02``
        for 2%).
    :type risk_pct: float
    :param entry_price: Intended entry price.
    :type entry_price: float
    :param stop_loss: Stop-loss price level.
    :type stop_loss: float
    :param min_quantity: Floor for the returned quantity. Defaults to ``1.0``.
    :type min_quantity: float
    :returns: Position size (number of units).
    :rtype: float
    :raises ValueError: If *entry_price* equals *stop_loss* (zero risk per
        unit makes sizing undefined).
    """
    risk_per_unit = abs(entry_price - stop_loss)
    if risk_per_unit == 0.0:
        raise ValueError("entry_price and stop_loss must differ")
    qty = (account_value * risk_pct) / risk_per_unit
    return max(qty, min_quantity)
