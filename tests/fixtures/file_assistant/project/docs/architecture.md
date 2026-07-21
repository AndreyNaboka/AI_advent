# Архитектура Store Demo

`OrderService` зависит от абстракции `PaymentGateway`. Прямое создание gateway
разрешено только в composition root и тестах.
