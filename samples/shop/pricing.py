"""Money is represented as integer cents in this fictional shop."""


def subtotal_cents(cart, prices):
    """Multiply each product's price by its quantity and sum the line amounts."""
    return sum(prices[sku] * quantity for sku, quantity in cart.items())


def discount_cents(subtotal, coupon):
    """WELCOME10 takes ten percent off; unknown coupons give no discount."""
    if coupon == "WELCOME10":
        return subtotal // 10
    return 0


def total_cents(cart, prices, coupon, shipping):
    """Apply the coupon to the subtotal, then add shipping without discounting it."""
    subtotal = subtotal_cents(cart, prices)
    return subtotal - discount_cents(subtotal, coupon) + shipping
