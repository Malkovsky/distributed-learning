from ..tcp_utils.protocol import *


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
