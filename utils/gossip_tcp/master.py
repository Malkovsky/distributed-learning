import numpy as np
import asyncio, sys
import pprint

import numpy.random

from ..tcp_utils.pickled_socket import PickledSocketWrapper
from ..tcp_utils.psocket_multiplexer import PSocketMultiplexer
from .protocol import *
from ..telemetry_processor import TelemetryProcessor


async def _assert_proto_ok(psocket, msg):
    resp = await psocket.recv()
    if not isinstance(resp, ProtoOk):
        print(msg)
        raise ProtoErrorException(msg + f': {resp!r}')


class GossipMaster:
    MASTER_TOKEN = 'MASTER'

    def __init__(self,
                 topology,
                 host: str, port: int,
                 debug=False,
                 telemetry_processor:TelemetryProcessor=None):
        self.gossip = self._GossipHandler(topology)
        self.agents: Dict[str, GossipMaster._AgentHandler] = dict()

        self.host: str = host
        self.port: int = port
        self.server: asyncio.Server = None

        self.debug = debug
        self.telemetry_processor = telemetry_processor

    async def serve_forever(self):
        print('topology info:')
        pprint.pprint(self.gossip.describe())

        self.server = await asyncio.start_server(self._handle_connection, self.host, self.port)
        print(f'Serving on {self.server.sockets[0].getsockname()}')
        try:
            async with self.server:
                await self.server.serve_forever()
        except Exception as e:  # todo: graceful shutdown
            print(f'Master: exception: {e}')
            print('Master: Shutting down...')
            for token in self.agents.keys():
                try:
                    print(f'Master: Sending ProtoShutdown to {token}... ', end='')
                    if self.agents[token].to_agent_psocket:
                        await self.agents[token].to_agent_psocket.send(ProtoShutdown())
                        print('ok')
                    else:
                        print('no outgoing connection to that agent!')
                except Exception as e:
                    print(f'error: {e}')
            raise

    def _debug(self, *args, **kwargs):
        if self.debug:
            if 'file' not in kwargs.keys():
                print('Master:', *args, **kwargs, file=sys.stderr)
            else:
                print('Master:', *args, **kwargs)

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        incoming_psocket = PickledSocketWrapper(reader, writer)
        data = await incoming_psocket.recv()
        remote_name = incoming_psocket.writer.get_extra_info("peername")
        if isinstance(data, ProtoShutdown):
            self._debug('Performing SHUTDOWN!')
            raise ProtoErrorException(data)  # todo: graceful shutdown
        if not isinstance(data, ProtoRegister):
            msg = f'From: {remote_name}. Expected registration data, got: {data!r}'
            self._debug(msg)
            await incoming_psocket.send(ProtoErrorException(msg))
            return
        if data.token not in self.gossip.tokens:
            msg = f'Agent {data.token} is not presented in topology'
            self._debug(f'From: {remote_name}.', msg)
            await incoming_psocket.send(ProtoErrorException(msg))
            return
        agent = self._AgentHandler(data.token)
        agent.from_agent_psocket = incoming_psocket
        agent.host = data.host
        agent.port = data.port
        self.agents[data.token] = agent
        self._debug(f'From: {remote_name}. Registered {data.token} -> {data.host, data.port}!')
        await incoming_psocket.send(ProtoOk())

        if len(self.agents) == len(self.gossip.tokens):
            await self._serve()

    async def _serve(self):
        self._debug('Starting to serve agents...')
        psockets_to_listen = {
            agent.token: agent.from_agent_psocket
            for agent in self.agents.values()
        }

        async for (token, req, psocket) in PSocketMultiplexer(psockets_to_listen):
            if isinstance(req, ProtoGossipNeighborRequest):
                self._debug(f'Received GossipNeighborRequest from {token}')
                resp = ProtoGossipNeighbor()
                resp.neighbor = self.gossip.get_matching_node_for_token(token, req.round_id)
                if resp.neighbor is not None:
                    resp.address = (self.agents[resp.neighbor].host, self.agents[resp.neighbor].port)
                await psocket.send(resp)
            elif isinstance(req, ProtoTelemetry):
                self._debug(f'Received Telemetry from {token}')
                await psocket.send(ProtoOk())
                try:
                    if self.telemetry_processor:
                        self.telemetry_processor.process(token, req.payload)
                except Exception as e:
                    print(f'Telemetry processor threw an exception: {e!r}')
            else:
                msg = f'Received unexpected request from {token}: {req!r}'
                print(msg)
                await psocket.send(ProtoErrorException(msg))

    class _AgentHandler:
        def __init__(self, token):
            self.token = token
            self.to_agent_psocket: PickledSocketWrapper = None
            self.from_agent_psocket: PickledSocketWrapper = None
            self.host = None
            self.port = None

    class _GossipHandler:
        def __init__(self, topology):
            self.topology = topology
            self.tokens = list(set(np.array(topology).flatten()))

            self.matchings = dict()

        def get_matching(self, round_id):
            if round_id not in self.matchings.keys():
                self.matchings[round_id] = self.__calc_matching(round_id)
            return self.matchings[round_id]

        def get_matching_node_for_token(self, token, round_id):
            if round_id not in self.matchings.keys():
                self.matchings[round_id] = self.__calc_matching(round_id)
            neighbor = None
            for (u, v) in self.matchings[round_id]:
                if u == token:
                    neighbor = v
                    break
                if v == token:
                    neighbor = u
                    break
            return neighbor

        def __calc_matching(self, round_id):  # greedy matching
            rnd = numpy.random.RandomState(round_id)
            permutation = list(range(len(self.topology)))
            rnd.shuffle(permutation)
            used = set()
            matching = []
            for (u, v) in [self.topology[i] for i in permutation]:
                if u not in used and v not in used:
                    used.add(u)
                    used.add(v)
                    matching.append((u, v))
            return matching

        def describe(self):
            return {
                'topology': self.topology,
            }
