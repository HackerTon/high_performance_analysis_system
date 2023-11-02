from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from typing import Union


class MetricPusher:
    def __init__(self, gateway_address: str) -> None:
        self.gateway_address = gateway_address
        self.registry = CollectorRegistry()
        self.person_gauge = Gauge(
            "number_of_person",
            documentation="Cumulative number of people detected",
            registry=self.registry,
        )
        self.timetaken_guage = Gauge(
            "latency",
            documentation="Latency for inference engine",
            registry=self.registry,
        )

    def push(self, number_of_person: Union[float, int], fps: Union[float, int]) -> None:
        self.person_gauge.set_to_current_time()
        self.timetaken_guage.set_to_current_time()
        self.person_gauge.set(number_of_person)
        self.timetaken_guage.set(fps)
        push_to_gateway(
            gateway=self.gateway_address,
            job="batch",
            registry=self.registry,
        )
