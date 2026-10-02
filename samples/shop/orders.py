"""Order transitions for a fictional shop; there is no payment integration."""


def create_order(cart, total):
    """An order starts pending and stores a copy of the cart and its total in cents."""
    if not cart:
        raise ValueError("Cannot create an empty order")
    return {"items": dict(cart), "total_cents": total, "status": "pending"}


def mark_paid(order):
    """Only pending orders can be marked paid; no payment is processed here."""
    if order["status"] != "pending":
        raise ValueError("Only pending orders can be marked paid")
    return {**order, "status": "paid"}


def cancel_order(order):
    """Only pending orders can be cancelled; paid orders need a separate refund flow."""
    if order["status"] != "pending":
        raise ValueError("Only pending orders can be cancelled")
    return {**order, "status": "cancelled"}
