import numpy as np
import asyncio
import sys
from enum import Enum

from .pickled_socket import PickledSocketWrapper
from .protocol import *
from .master import ConsensusMaster, _assert_proto_ok


class ConsensusAgent:
    class Status(Enum):
        INIT = 1,
        REGISTRATION_COMPLETE = 2,
        GOT_NEIGHBORHOOD_DATA = 3,
        NETWORK_READY = 4,
        RUNNING_CONSENSUS = 5

        def __lt__(self, other):
            if not isinstance(other, ConsensusAgent.Status):
                raise ValueError(f'Cannot compare Status and {other!r}')
            return self.value < other.value

    def __init__(self,
                 token,
                 host, port,
                 master_host, master_port,
                 public_host_port=None,
                 debug=False):
        self.token = token
        self.host = host
        self.port = port
        self.public_host_port = (host, port) if public_host_port is None else public_host_port
        self.server: asyncio.Server = None

        self.master = self._MasterHandler()
        self.master.host, self.master.port = master_host, master_port

        self.neighbors: Dict[str, ConsensusAgent._NeighborAgentHandler] = dict()

        self.debug = debug

        self.status = self.Status.INIT
        self.value = None  # not None only while self.status == Status.RUNNING_CONSENSUS

    def _debug(self, *args, **kwargs):
        if self.debug:
            if 'file' not in kwargs.keys():
                print(f'Agent {self.token}:', *args, **kwargs, file=sys.stderr)
            else:
                print(f'Agent {self.token}:', *args, **kwargs)

    async def serve_forever(self):
        self.server = await asyncio.start_server(self._handle_connection, self.host, self.port)
        print(f'Agent {self.token}: Serving on {self.server.sockets[0].getsockname()}')
        try:
            async with self.server:
                await self.server.serve_forever()
        except Exception as e:  # todo: graceful shutdown
            print(f'Agent {self.token}: Error happened while serving: {e!r}')
            raise e

    async def _handle_connection(self, reader, writer):
        psocket = PickledSocketWrapper(reader, writer)
        data = await psocket.recv()
        remote_name = psocket.writer.get_extra_info("peername")
        if not isinstance(data, ProtoRegister):
            msg = f'Agent {self.token}: From: {remote_name}. Expected registration data, got: {data!r}'
            print(msg)
            await psocket.send(ProtoErrorException(msg))
            return
        if data.token == ConsensusMaster.MASTER_TOKEN:
            if self.status > self.Status.INIT:
                msg = f'Agent {self.token}: Post-init master connection: illegal state'
                print(msg)
                await psocket.send(ProtoErrorException(msg))
                return
            self._debug('Got incoming connection from master!')
            self.master.from_master_psocket = psocket
            await psocket.send(ProtoOk())

            self._debug(f'Waiting for neighborhood data')
            # now master will send us neighborhood data
            neighborhood_data = await psocket.recv()
            if not isinstance(neighborhood_data, ProtoNeighborhoodData):
                msg = f'Agent {self.token}: Expected neighborhood data, got: {neighborhood_data!r}'
                print(msg)
                await psocket.send(ProtoErrorException(msg))
                return
            # self.convergence_rate = neighborhood_data.global_convergence_rate
            self._debug('Got neighborhood data from master: ', neighborhood_data)
            for token, (host, port) in neighborhood_data.neighbors.items():
                neighbor = self._NeighborAgentHandler(token)
                neighbor.host, neighbor.port = host, port
                neighbor.edge_weight = neighborhood_data.weights[token]
                self.neighbors[token] = neighbor
            await psocket.send(ProtoOk())
            self._debug('Changing status to GOT_NEIGHBORHOOD_DATA')
            self.status = self.Status.GOT_NEIGHBORHOOD_DATA  # trigger sleeping coros
        else:
            # we need to verify that this incoming connection is from our neighbor
            # we cannot do that until master tells us who is our neighbor
            while self.status < self.Status.GOT_NEIGHBORHOOD_DATA:
                await asyncio.sleep(0.05)

            if data.token not in self.neighbors.keys(): # unknown neighbor / connection
                msg = f'Agent {self.token}: Incoming connection from {data.token}: not in my neighborhood'
                print(msg)
                await psocket.send(ProtoErrorException(msg))
                return

            self._debug(f'Got handshake from neighbor {data.token}')
            self.neighbors[data.token].from_neighbor_psocket = psocket
            await psocket.send(ProtoOk())

    async def _do_handshake(self):
        """
        Performs handshake with master and all neighbors
        After completion, self.status will be Status.NETWORK_READY
        """
        self._debug('Performing handshake with master')
        reader, writer = await asyncio.open_connection(self.master.host, self.master.port)
        self.master.to_master_psocket = PickledSocketWrapper(reader, writer)
        await self.master.to_master_psocket.send(
            ProtoRegister(self.token, self.public_host_port[0], self.public_host_port[1])
        )
        await _assert_proto_ok(self.master.to_master_psocket,
                               f'Agent {self.token}: Master handshake: Unexpected response')
        self._debug('Successfully registered')
        # when all agents pass this point,
        # master will open new connections to all agents and send them neighborhood data.
        # handle_connection will process master's info, we just need to wait until initialization completes
        while self.status < self.Status.GOT_NEIGHBORHOOD_DATA:
            await asyncio.sleep(0.05)
        self._debug('do_handshake GOT_NEIGHBORHOOD_DATA trigger')

        for neighbor in self.neighbors.values():
            self._debug(f'Performing handshake with {neighbor.token}')
            n_reader, n_writer = await asyncio.open_connection(neighbor.host, neighbor.port)
            neighbor.to_neighbor_psocket = PickledSocketWrapper(n_reader, n_writer)

            await neighbor.to_neighbor_psocket.send(ProtoRegister(self.token, self.host, self.port))
            await _assert_proto_ok(neighbor.to_neighbor_psocket,
                                   f'Agent {self.token}: handshake with neighbor {neighbor.token} has failed')

        # we have successfully set up outgoing connections to our neighbors
        # we need to wait until they open connections to us
        self._debug(f'Waiting for incoming connections from neighbors...')
        while not all([neighbor.from_neighbor_psocket is not None for neighbor in self.neighbors.values()]):
            await asyncio.sleep(0.05)
        self._debug(f'We are ready for consensus now!')
        self._debug(f'Changing status to NETWORK_READY')
        self.status = self.Status.NETWORK_READY

    async def run_round(self, value, weight, convergence_eps=None):
        raise NotImplemented()  # :)

    async def run_once(self, value):
        """
        Runs one iteration of consensus
        This method waits until all our neighbors got our value
        """
        if self.status < self.Status.NETWORK_READY:
            await self._do_handshake()
        if self.status == self.Status.RUNNING_CONSENSUS:
            msg = 'Consensus is already running! Illegal state?'
            print(msg)
            raise AssertionError(msg)

        self.status = self.Status.RUNNING_CONSENSUS
        self._debug(f'consensus iteration start')
        self.value = value

        async def respond_neighbor_once(neighbor_token):
            psocket = self.neighbors[neighbor_token].from_neighbor_psocket
            req = await psocket.recv()
            if not isinstance(req, ProtoRunOnceValueRequest):
                msg = f'Agent {self.token}: run_once: unexpected request from neighbor {neighbor_token}: {req!r}'
                print(msg)
                await psocket.send(ProtoErrorException(msg))
                return
            self._debug(f'sending value to {neighbor_token}')
            await psocket.send(ProtoRunOnceValueResponse(self.value))

        neighbor_values = {}

        async def ask_neighbor_once(neighbor):
            self._debug(f'requesting value from neighbor {neighbor.token}')
            await neighbor.to_neighbor_psocket.send(ProtoRunOnceValueRequest())
            resp = await neighbor.to_neighbor_psocket.recv()
            if not isinstance(resp, ProtoRunOnceValueResponse):
                msg = f'Agent {self.token}: run_once: unexpected response from neighbor {neighbor.token}: {resp!r}'
                print(msg)
                raise ProtoErrorException(msg)
            neighbor_values[neighbor.token] = resp.value

        respond_tasks = [asyncio.create_task(respond_neighbor_once(neighbor.token))
                         for neighbor in self.neighbors.values()]
        ask_tasks = [asyncio.create_task(ask_neighbor_once(neighbor))
                     for neighbor in self.neighbors.values()]
        await asyncio.gather(*(respond_tasks + ask_tasks))

        # we've got all the values from neighbors, now we can update our value
        sum_neighbor_weights = np.sum([neighbor.edge_weight for neighbor in self.neighbors.values()])
        value = (1.0 - sum_neighbor_weights) * value + \
                np.sum([neighbor_values[token] * self.neighbors[token].edge_weight
                        for token in self.neighbors.keys()], axis=0)
        self._debug(f'final result: {value}')
        self.value = None
        self._debug(f'consensus iteration end')
        self.status = self.Status.NETWORK_READY
        return value

    async def send_telemetry(self, payload):
        if self.status < self.Status.NETWORK_READY:
            await self._do_handshake()
        await self.master.to_master_psocket.send(ProtoTelemetry(payload))
        await _assert_proto_ok(self.master.to_master_psocket, "Expected to receive ProtoOk from master")

    class _MasterHandler:
        def __init__(self):
            self.token = ConsensusMaster.MASTER_TOKEN
            self.host = None
            self.port = None
            self.to_master_psocket: PickledSocketWrapper = None
            self.from_master_psocket: PickledSocketWrapper = None

    class _NeighborAgentHandler:
        def __init__(self, token):
            self.token = token
            self.host = None
            self.port = None
            self.to_neighbor_psocket: PickledSocketWrapper = None
            self.from_neighbor_psocket: PickledSocketWrapper = None

            self.edge_weight = None
