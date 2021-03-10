import numpy as np
import asyncio, sys
from typing import Dict, List, Any
import pprint

from .pickled_socket import PickledSocketWrapper
from .protocol import *
from .psocket_multiplexer import PSocketMultiplexer

from ..fast_averaging import find_optimal_weights


class _AgentData:
    def __init__(self, token):
        self.token = token
        self.to_agent_psocket: PickledSocketWrapper = None
        self.from_agent_psocket: PickledSocketWrapper = None
        self.host = None
        self.port = None

        self.asked_for_new_round = False
        self.weight = 0.0
        self.has_converged = False


class _ConsensusData:
    def __init__(self, topology):
        self.topology = topology
        self.tokens = list(set(np.array(topology).flatten()))
        self.edge_weights = None
        self.convergence_rate = None

        self.round_counter = 0
        self.round_is_running = False

    def get_neighborhood_info_for_agent(self, token):
        self._solve_fastest_convergence()

        neighbors = {
            u if token == v else v
            for (u, v) in self.topology
            if token == u or token == v
        }
        topology_w_edge_weights = zip(self.topology, self.edge_weights)
        agent_top_ew = filter(lambda uv_c: (uv_c[0][0] == token or uv_c[0][1] == token), topology_w_edge_weights)
        agent_edge_weights = {
            token: [c for ((u, v), c) in agent_top_ew if (u == token or v == token)][0]
            for token in neighbors
        }

        return agent_edge_weights, self.convergence_rate

    def describe(self):
        E = np.array([ [
                int((u, v) in self.topology or (v, u) in self.topology)
                for v in self.tokens
            ] for u in self.tokens])
        outdeg = np.sum(E, axis=1)
        L = np.diag(outdeg) - E
        L_eig = np.linalg.eigvals(L)
        L_eig.sort()
        self._solve_fastest_convergence()
        return {
            'laplacian': L,
            'eigenvalues': L_eig,
            'algebraic_connectivity': L_eig[1],
            'convergence_speed': self.convergence_rate
        }

    def _solve_fastest_convergence(self):
        if self.edge_weights is not None:
            return self.edge_weights, self.convergence_rate
        self.edge_weights, self.convergence_rate = find_optimal_weights(self.topology)
        return self.edge_weights, self.convergence_rate


async def _assert_proto_ok(psocket, msg):
    resp = await psocket.recv()
    if not isinstance(resp, ProtoOk):
        print(msg)
        raise ProtoErrorException(msg + f': {resp!r}')


class ConsensusMaster:
    MASTER_TOKEN = 'MASTER'

    def __init__(self,
                 topology,
                 host: str, port: int,
                 debug=False):
        self.consensus = _ConsensusData(topology)
        self.agents: Dict[str, _AgentData] = dict()

        self.host: str = host
        self.port: int = port
        self.server: asyncio.Server = None

        self.debug = debug

    async def serve_forever(self):
        print('topology info:')
        pprint.pprint(self.consensus.describe())

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
        if data.token not in self.consensus.tokens:
            msg = f'Agent {data.token} is not presented in topology'
            self._debug(f'From: {remote_name}.', msg)
            await incoming_psocket.send(ProtoErrorException(msg))
            return
        agent = _AgentData(data.token)
        agent.from_agent_psocket = incoming_psocket
        agent.host = data.host
        agent.port = data.port
        self.agents[data.token] = agent
        self._debug(f'From: {remote_name}. Registered {data.token} -> {data.host, data.port}!')
        await incoming_psocket.send(ProtoOk())

        if len(self.agents) == len(self.consensus.tokens):
            self._debug('All agents have registered! Starting initialization procedure...')
            await self._initialize_agents()

    async def _initialize_agents(self):
        for agent in self.agents.values():
            try:
                self._debug(f'Opening outgoing connection to {agent.token}')
                reader, writer = await asyncio.open_connection(agent.host, agent.port)
                agent.to_agent_psocket = PickledSocketWrapper(reader, writer)

                self._debug(f'Sending registration data to {agent.token}')
                await agent.to_agent_psocket.send(ProtoRegister(ConsensusMaster.MASTER_TOKEN, self.host, self.port))
                await _assert_proto_ok(agent.to_agent_psocket,
                                       f'master -> agent registration response is not ProtoOk')
            except Exception as e:
                msg = f'Master: Failed to open outgoing connection to {agent.token}: {e}'
                print(msg)
                raise e

            edge_weights, convergence_rate = self.consensus.get_neighborhood_info_for_agent(agent.token)
            neighbors = {
                token: (self.agents[token].host, self.agents[token].port)
                for token in edge_weights.keys()
            }

            self._debug(f'Sending neighborhood data to {agent.token}')
            await agent.to_agent_psocket.send(ProtoNeighborhoodData(neighbors, edge_weights, convergence_rate))
            await _assert_proto_ok(agent.to_agent_psocket,
                                   f'Master: Send neighborhood data: Got unexpected response from {agent.token}')
        print('Master: All agents have been initialized!')
        await self._serve()
            
    async def _serve(self):
        self._debug('Starting to serve agents...')
        psockets_to_listen = {
            agent.token: agent.from_agent_psocket
            for agent in self.agents.values()
        }
        self.consensus.round_is_running = False
        self.consensus.round_counter = 0

        async for (token, req, psocket) in PSocketMultiplexer(psockets_to_listen):
            if isinstance(req, ProtoNewRoundRequest):
                self._debug(f'Received NEW_ROUND_REQ from {token} with weight {req.weight}')
                if self.running_round:
                    msg = f'Master: Received NEW_ROUND_REQ from {token} but round is already running'
                    print(msg)
                    await psocket.send(ProtoErrorException(msg))
                    continue
                self.agents[token].asked_for_new_round = True
                self.agents[token].weight = req.weight
                await psocket.send(ProtoOk())

                if all([agent.asked_for_new_round for agent in self.agents.values()]):
                    self.consensus.round_counter += 1
                    self.consensus.round_is_running = True
                    self._debug(f'===== STARTING A NEW ROUND {self.consensus.round_counter} =====')

                    for agent in self.agents.values():
                        agent.asked_for_new_round = False
                        agent.has_converged = False

                    for agent in self.agents.values():  # todo: run concurrently through asyncio.gather/wait
                        self._debug(f'Sending new round notification to {agent.token}')
                        await agent.to_agent_psocket.send(ProtoNewRoundNotification(self.consensus.round_counter))
                        await _assert_proto_ok(agent.to_agent_psocket,
                                               f'Master: New round notification: '
                                               f'Got unexpected response from {agent.token}')
            elif isinstance(req, ProtoConverged):
                self._debug(f'Received CONVERGED from {token}')
                if not self.running_round:
                    msg = f'Received CONVERGED from {token} but round is not yet running'
                    print(msg)
                    await psocket.send(ProtoErrorException(msg))
                    continue
                self.agents[token].has_converged = True
                await psocket.send(ProtoOk())

                if all([agent.has_converged for agent in self.agents.values()]):
                    self._debug('===== ALL NODES CONVERGED! DONE =====')
                    self.running_round = False
                    for agent in self.agents.values():  # todo: run concurrently through asyncio.gather/wait
                        self._debug(f'Sending DONE notification to {agent.token}')
                        await agent.to_agent_psocket.send(ProtoDone(self.consensus.round_counter))
                        await _assert_proto_ok(agent.to_agent_psocket,
                                               f'Master: Round DONE notification: '
                                               f'Got unexpected response from {agent.token}')
            elif isinstance(req, ProtoNotConverged):
                self._debug(f'Received NOT_CONVERGED from {token}')
                if not self.running_round:
                    msg = f'got NOT_CONVERGED from {token} but round is not yet running'
                    print(msg)
                    await psocket.send(ProtoErrorException(msg))
                    continue
                self.agents[token].has_converged = False
                await psocket.send(ProtoOk())
            elif isinstance(req, ProtoTelemetry):
                self._debug(f'Received Telemetry from {token} with tag {req.tag}')
                # TODO
                self._debug(f'Telemetry is not yet supported')
                await psocket.send(ProtoOk())
            else:
                msg = f'Received unexpected request from {token}: {req!r}'
                print(msg)
                await psocket.send(ProtoErrorException(msg))
