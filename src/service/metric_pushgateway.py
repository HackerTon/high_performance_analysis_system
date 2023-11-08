from typing import Union

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from urllib.error import URLError


class MetricPusher:
    def __init__(self, gateway_address: str) -> None:
        self.gateway_address = gateway_address
        self.registry = CollectorRegistry()
        self.person_right_to_left = Gauge(
            "person_right_to_left",
            documentation="Cumulative person crossing from right to left",
            registry=self.registry,
        )
        self.person_left_to_right = Gauge(
            "person_left_to_right",
            documentation="Cumulative person crossing from left to right",
            registry=self.registry,
        )
        self.timetaken_guage = Gauge(
            "latency",
            documentation="Latency for inference engine",
            registry=self.registry,
        )

    def push(
        self,
        person_left_to_right: int,
        person_right_to_left: int,
        latency: Union[float, int],
    ) -> None:
        self.person_right_to_left.set_to_current_time()
        self.person_left_to_right.set_to_current_time()
        self.timetaken_guage.set_to_current_time()
        self.person_left_to_right.set(person_left_to_right)
        self.person_right_to_left.set(person_right_to_left)
        self.timetaken_guage.set(latency)
        try:
            push_to_gateway(
                gateway=self.gateway_address,
                job="batch",
                registry=self.registry,
            )
        except URLError as error:
            print(error.reason)
