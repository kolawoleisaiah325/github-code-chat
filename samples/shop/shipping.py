"""Delivery fees for a fictional shop, expressed in integer cents."""


def shipping_fee_cents(country, subtotal):
    """Domestic shipping is free at a subtotal of 5000 cents, otherwise 500 cents."""
    if country == "NG":
        return 0 if subtotal >= 5000 else 500
    return 2000


def delivery_days(country):
    """Estimated delivery takes three days domestically and ten internationally."""
    return 3 if country == "NG" else 10
