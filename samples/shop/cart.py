"""A small shopping cart used only as an inspectable demonstration corpus."""


def add_item(cart, sku, quantity):
    """Add a positive integer quantity to a product's existing cart quantity."""
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 1:
        raise ValueError("Quantity must be a positive integer")
    updated = dict(cart)
    updated[sku] = updated.get(sku, 0) + quantity
    return updated


def remove_item(cart, sku):
    """Remove a product entirely; removing an absent product leaves the cart unchanged."""
    updated = dict(cart)
    updated.pop(sku, None)
    return updated


def count_units(cart):
    """Count all item units, rather than the number of distinct products."""
    return sum(cart.values())
