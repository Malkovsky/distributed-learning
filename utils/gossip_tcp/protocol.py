from ..tcp_utils.protocol import *


@dataclass
class ProtoGossipNeighborRequest:
    round_id: int


@dataclass
class ProtoGossipNeighbor:
    neighbor: str  # might be None
    address: Tuple[str, int]


@dataclass
class ProtoGossipValueExchange:
    round_id: int
    value: Any
