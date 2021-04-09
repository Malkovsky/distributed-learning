from dataclasses import dataclass
from typing import List, Tuple, Any, Dict, Set

'''
Protocol communication invariants:
* Both master and agent act as server and client.
I.e. agent opens connection to master, sends it requests and receives responses to those requests and only those requests.
When master needs to send some data that agent didn't request, master opens its own connection to the agent, 
so their roles as server and client are swapped.
* All requests are responded with exactly one response object. It will be ProtoOk() most of the time.
But in case of an error, the response might be an instance of ProtoErrorException.
'''


class ProtoErrorException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f'Consensus Protocol Error: {self.msg}'


@dataclass
class ProtoRegister:
    token: str
    host: str
    port: int


@dataclass
class ProtoOk:
    pass


@dataclass
class ProtoNeighborhoodData:
    neighbors: Dict[str, Tuple[str, int]]   # (token -> (host, port))
    weights: Dict[str, float]
    global_convergence_rate: float


@dataclass
class ProtoConverged:
    pass


@dataclass
class ProtoNotConverged:
    pass


@dataclass
class ProtoNewRoundRequest:
    weight: float


@dataclass
class ProtoNewRoundNotification:
    round_id: int


@dataclass
class ProtoRunOnceValueRequest:
    pass


@dataclass
class ProtoRunOnceValueResponse:
    value: Any


@dataclass
class ProtoDone:
    round_id: int


@dataclass
class ProtoShutdown:
    pass


@dataclass
class ProtoTelemetry:
    payload: Any
