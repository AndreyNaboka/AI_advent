"""Application service coordinating orders and payments."""

from app.payment_api import PaymentGateway


class OrderService:
    def __init__(self, gateway: PaymentGateway):
        self.gateway = gateway

    def checkout(self, total: float) -> str:
        return self.gateway.charge(total)
