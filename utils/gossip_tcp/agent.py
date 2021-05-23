import numpy as np
import asyncio
import sys
from enum import Enum

from ..tcp_utils.pickled_socket import PickledSocketWrapper
from .protocol import *
from .master import GossipMaster, _assert_proto_ok


class GossipAgent:
    class Status(Enum):
        INIT = 1
        REGISTRATION_COMPLETE = 2
        NETWORK_READY = 3
        RUNNING_GOSSIP = 4

        def __lt__(self, other):
            if not isinstance(other, GossipAgent.Status):
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

        self.neighbors: Dict[str, GossipAgent._NeighborAgentHandler] = dict()

        self.debug = debug

        self.status = self.Status.INIT
        self.round_id = 0  # increments at the beginning of run_once

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
        if data.token == GossipMaster.MASTER_TOKEN:
            if self.status > self.Status.INIT:
                msg = f'Agent {self.token}: Post-init master connection: illegal state'
                print(msg)
                await psocket.send(ProtoErrorException(msg))
                return
            self._debug('Got incoming connection from master!')
            self.master.from_master_psocket = psocket
            await psocket.send(ProtoOk())

            self._debug('Changing status to NETWORK_READY')
            self.status = self.Status.NETWORK_READY  # trigger sleeping coros
        else:
            # we need to verify that this incoming connection is from our neighbor
            # we cannot do that until master tells us who is our neighbor
            while self.status < self.Status.NETWORK_READY:
                await asyncio.sleep(0.05)

            self._debug(f'Got handshake from node {data.token}')
            if data.token not in self.neighbors.keys():
                self.neighbors[data.token] = self._NeighborAgentHandler(data.token)
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

        self._debug(f'We are ready for gossip now!')
        self._debug(f'Changing status to NETWORK_READY')
        self.status = self.Status.NETWORK_READY

    async def run_once(self, value):
        """
        Runs one iteration of gossip
        This method waits until our neighbor on current iteration got our value
        """
        if self.status < self.Status.NETWORK_READY:
            await self._do_handshake()
        if self.status == self.Status.RUNNING_GOSSIP:
            msg = 'Gossip is already running! Illegal state?'
            print(msg)
            raise AssertionError(msg)

        self.round_id += 1
        self.status = self.Status.RUNNING_GOSSIP
        self._debug(f'gossip {self.round_id} iteration start')
        self.value = value

        self._debug(f'asking master')
        await self.master.to_master_psocket.send(ProtoGossipNeighborRequest(self.round_id))
        resp = await self.master.to_master_psocket.recv()

        if not isinstance(resp, ProtoGossipNeighbor):
            msg = f'Agent {self.token}: run_once: unexpected response from master: {resp!r}'
            print(msg)
            raise ProtoErrorException(msg)

        if resp.neighbor is None:
            self._debug(f'no neighbor for round {self.round_id}')
            self.status = self.Status.NETWORK_READY
            return value

        neighbor_token = resp.neighbor
        neighbor_address = resp.address  # (host, port)

        async def listen_to_neighbor(neighbor_token):
            while neighbor_token not in self.neighbors.keys() or self.neighbors[neighbor_token].from_neighbor_psocket is None:
                await asyncio.sleep(0.05)
            psocket = self.neighbors[neighbor_token].from_neighbor_psocket
            req = await psocket.recv()
            if not isinstance(req, ProtoGossipValueExchange):
                msg = f'Agent {self.token}: run_once: unexpected data from neighbor {neighbor_token}: {req!r}'
                print(msg)
                await psocket.send(ProtoErrorException(msg))
                raise ProtoErrorException(msg)
            if req.round_id != self.round_id:
                msg = f'Agent {self.token}: run_once: got unexpected round_id from neighbor {neighbor_token}: {req!r}, expected {self.round_id}'
                print(msg)
                await psocket.send(ProtoErrorException(msg))
                raise ProtoErrorException(msg)
            self._debug(f'got value from {neighbor_token}')
            await psocket.send(ProtoOk())
            return req.value

        async def send_neighbor_our_value(neighbor, address):
            if neighbor not in self.neighbors.keys():
                handler = self._NeighborAgentHandler(neighbor)
                handler.host = address[0]
                handler.port = address[1]
                self.neighbors[neighbor] = handler
            if self.neighbors[neighbor].to_neighbor_psocket is None:
                handler = self.neighbors[neighbor]
                reader, writer = await asyncio.open_connection(address[0], address[1])
                handler.to_neighbor_psocket = PickledSocketWrapper(reader, writer)
                await handler.to_neighbor_psocket.send(
                    ProtoRegister(self.token, self.public_host_port[0], self.public_host_port[1])
                )
                await _assert_proto_ok(handler.to_neighbor_psocket,
                                       f'Agent {self.token}: Neighbor handshake: Unexpected response')
                self._debug(f'Successfully connected to neighbor {neighbor_token}')


            self._debug(f'sending our value to {neighbor_token}')
            await self.neighbors[neighbor].to_neighbor_psocket.send(ProtoGossipValueExchange(self.round_id, value))
            resp = await self.neighbors[neighbor].to_neighbor_psocket.recv()
            if not isinstance(resp, ProtoOk):
                msg = f'Agent {self.token}: run_once: unexpected response from neighbor {neighbor_token}: {resp!r}'
                print(msg)
                raise ProtoErrorException(msg)

        listen_task = [asyncio.create_task(listen_to_neighbor(neighbor_token))]
        send_task = [asyncio.create_task(send_neighbor_our_value(neighbor_token, neighbor_address))]
        await asyncio.gather(*(listen_task + send_task))

        neighbor_value = listen_task[0].result()
        value = (value + neighbor_value) / 2.0
        self._debug(f'final result: {value}')
        self._debug(f'gossip iteration end')
        self.status = self.Status.NETWORK_READY
        return value

    async def send_telemetry(self, payload):
        if self.status < self.Status.NETWORK_READY:
            await self._do_handshake()
        await self.master.to_master_psocket.send(ProtoTelemetry(payload))
        await _assert_proto_ok(self.master.to_master_psocket, "Expected to receive ProtoOk from master")

    class _MasterHandler:
        def __init__(self):
            self.token = GossipMaster.MASTER_TOKEN
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
