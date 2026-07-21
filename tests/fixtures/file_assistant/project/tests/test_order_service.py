from app.order_service import OrderService
from app.payment_api import PaymentGateway


def test_checkout_uses_gateway():
    service = OrderService(PaymentGateway())
    assert service.checkout(10) == "payment-demo-id"
