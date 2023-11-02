from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from typing import Union


class MetricPusher:
    def __init__(self, gateway_address: str) -> None:
        self.gateway_address = gateway_address
        self.registry = CollectorRegistry()
        self.gauge = Gauge(
            "number_of_person", documentation="Cumulative number of people detected", registry=self.registry
        )

    def push(self, number_of_person: Union[float, int]) -> None:
        self.gauge.set_to_current_time()
        self.gauge.set(number_of_person)
        push_to_gateway(
            gateway=self.gateway_address,
            job="batcha",
            registry=self.registry,
        )
