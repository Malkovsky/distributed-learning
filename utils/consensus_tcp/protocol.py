from dataclasses import dataclass
from typing import List, Tuple, Any, Dict

'''
Master-Agent communication:
Registration:
    Agent  ---Register(token)--> Master
    Agent  <------OK------------ Master
Post-registration (all agents registered):
    Agent  <--NeighborhoodData-- Master
    Agent  -------OK-----------> Master
    Agent  <---NeighborWeights-- Master
    Agent  -------OK-----------> Master
TODO
    
Agent-Agent communication:
Every agent acts both like server (responding to requests) and client (performing requests).

Registration:
    Client  ---Register(token)--> Server
    Client  <------OK------------ Server
TODO
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
    neighbors: List[Tuple[str, str, int]]   # (token, host, port)


@dataclass
class ProtoNeighborWeights:
    weights: Dict[str, float]
    convergence_rate: float


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
    mean_weight: float


@dataclass
class ProtoValueRequest:
    round_id: int
    round_iteration: int


@dataclass
class ProtoValueResponse:
    round_id: int
    round_iteration: int
    value: Any


@dataclass
class ProtoDone:
    round_id: int


@dataclass
class ProtoShutdown:
    pass