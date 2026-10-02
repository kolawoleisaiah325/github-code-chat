"""In-memory inventory rules for a fictional shop."""


def available_stock(stock, reserved, sku):
    """Available stock subtracts reservations, never returning a negative quantity."""
    return max(0, stock.get(sku, 0) - reserved.get(sku, 0))


def check_inventory(cart, stock, reserved):
    """Return the product identifiers with insufficient available stock for the cart."""
    return [sku for sku, quantity in cart.items()
            if quantity > available_stock(stock, reserved, sku)]


def reserve_inventory(cart, stock, reserved):
    """Reject an out-of-stock cart before reserving any product."""
    shortages = check_inventory(cart, stock, reserved)
    if shortages:
        raise ValueError("Insufficient stock: " + ", ".join(shortages))
    updated = dict(reserved)
    for sku, quantity in cart.items():
        updated[sku] = updated.get(sku, 0) + quantity
    return updated
