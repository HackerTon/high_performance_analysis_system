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
        self.frame_left_gauge = Gauge(
            "frame_left",
            documentation="Frame left",
            registry=self.registry,
        )

    def push(
        self,
        number_of_person: Union[float, int],
        latency: Union[float, int],
        frame_left: int,
    ) -> None:
        print('update')
        self.person_gauge.set_to_current_time()
        self.timetaken_guage.set_to_current_time()
        self.frame_left_gauge.set_to_current_time()
        self.person_gauge.set(number_of_person)
        self.timetaken_guage.set(latency)
        self.frame_left_gauge.set(frame_left)
        push_to_gateway(
            gateway=self.gateway_address,
            job="batch",
            registry=self.registry,
        )
