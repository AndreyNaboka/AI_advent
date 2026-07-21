"""Payment gateway abstraction and a small default implementation."""


class PaymentGateway:
    def charge(self, amount: float) -> str:
        if amount <= 0:
            raise ValueError("amount must be positive")
        return "payment-demo-id"


def build_gateway() -> PaymentGateway:
    return PaymentGateway()
