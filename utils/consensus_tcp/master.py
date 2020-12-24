import numpy as np
import asyncio, sys
from typing import Dict, List, Any

from .pickled_socket import PickledSocketWrapper
from .protocol import *
from .psocket_selector import PSocketSelector

from ..fast_averaging import find_optimal_weights

class ConsensusMaster:
    def __init__(self, topology, host: str, port: int, debug=False):
        self.topology = topology
        self.tokens = list(set(np.array(topology).flatten()))
        self.edge_weights = None
        self.convergence_rate = None

        self.host: str = host
        self.port: int = port
        self.server: asyncio.Server = None

        self.agent_psockets: Dict[Any, PickledSocketWrapper] = dict()
        self.agent_host: Dict[Any, str] = dict()
        self.agent_port: Dict[Any, int] = dict()
        self.agent_psocket_selector = PSocketSelector()

        self.round_counter = 0
        self.running_round = False
        self.agent_new_round = dict()
        self.agent_weight = dict()
        self.agent_converged = dict()
        
        self.debug = debug

    async def serve_forever(self):
        self.server = await asyncio.start_server(self.handle_connection, self.host, self.port)
        print(f'Serving on {self.server.sockets[0].getsockname()}')
        try:
            async with self.server:
                serve_task = self.server.get_loop().create_task(self._serve())
                await self.server.serve_forever()
        except:
            serve_task.cancel()
            for token in self.tokens:
                try:
                    await self.agent_psockets[token].send(ProtoShutdown())
                except:
                    pass
            raise

    def _debug(self, *args, **kwargs):
        if self.debug:
            if 'file' not in kwargs.keys():
                print('Master:', *args, **kwargs, file=sys.stderr)
            else:
                print('Master:', *args, **kwargs)
    
    def describe(self):
        E = np.array([ [
                int((u, v) in self.topology or (v, u) in self.topology)
                for v in self.tokens
            ] for u in self.tokens])
        outdeg = np.sum(E, axis=1)
        L = np.diag(outdeg) - E
        print('Laplacian:\n{}'.format(L))
        L_eig = np.linalg.eigvals(L)
        L_eig.sort()
        print('Eigenvalues: {}'.format(L_eig))
        print('Algebraic connectivity: {}'.format(L_eig[1]))
        self._solve_fastest_convergence()
        print('Convergence speed: {}'.format(self.convergence_rate))

    def _solve_fastest_convergence(self):
        if self.edge_weights is not None:
            return self.edge_weights, self.convergence_rate
        self.edge_weights, self.convergence_rate = find_optimal_weights(self.topology)
        return self.edge_weights, self.convergence_rate

    async def handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        psocket = PickledSocketWrapper(reader, writer)
        data = await psocket.recv()
        remote_name = psocket.writer.get_extra_info("peername")
        if isinstance(data, ProtoShutdown):
            self._debug('Performing SHUTDOWN!')
            raise ProtoErrorException(data)
        if not isinstance(data, ProtoRegister):
            msg = f'From: {remote_name}. Expected registration data, got: {data!r}'
            self._debug(msg)
            raise ProtoErrorException(msg)
        self.agent_psockets[data.token] = psocket
        self.agent_host[data.token] = data.host
        self.agent_port[data.token] = data.port
        self._debug(f'From: {remote_name}. Registered {data.token} -> {data.host, data.port}!')
        await psocket.send(ProtoOk())

        if len(self.agent_psockets) == len(self.tokens):
            self._debug('All agents registered! Starting initialization procedure...')
            await self.initialize_agents()

    async def initialize_agents(self):
        for token, psocket in self.agent_psockets.items():
            neighbors = [u if token == v else v for (u, v) in self.topology if token == u or token == v]
            data = [(n_token, self.agent_host[n_token], self.agent_port[n_token])
                    for n_token in neighbors]

            self._debug(f'Sending neighborhood data to {token}')
            await psocket.send(ProtoNeighborhoodData(data))
            self._debug(f'Waiting for OK...')
            resp = await psocket.recv()
            if not isinstance(resp, ProtoOk):
                msg = f'Got unexpected response from {token}: {resp!r}'
                self._debug(msg)
                raise ProtoErrorException(msg)
            self._debug(f'Got OK from {token}')

            self._solve_fastest_convergence()
            self._debug(f'Sending neighbor weights to {token}')
            weights = {}
            for (edge, weight) in zip(self.topology, self.edge_weights):
                n = None
                if edge[0] == token:
                    n = edge[1]
                if edge[1] == token:
                    n = edge[0]
                if n is not None:
                    weights[n] = weight
            await psocket.send(ProtoNeighborWeights(weights, self.convergence_rate))
            self._debug(f'Waiting for OK...')
            resp = await psocket.recv()
            if not isinstance(resp, ProtoOk):
                msg = f'Got unexpected response from {token}: {resp!r}'
                self._debug(msg)
                raise ProtoErrorException(msg)
            self._debug(f'Got OK from {token}')

            self.agent_psocket_selector.add(token, psocket)
        self._debug('All agents have been initialized!')
            
    async def _serve(self):
        self._debug('Serving...')
        self.agent_new_round = { token: False for token in self.tokens }
        self.agent_converged = { token: False for token in self.tokens }

        while True:
            token, req = await self.agent_psocket_selector.recv(0.5)
            if token is None:
                await asyncio.sleep(0.25)
                continue
            self._debug(f'Got from {token}: {req!r}')
            if isinstance(req, ProtoNewRoundRequest):
                self._debug(f'got NEW_ROUND from "{token}" with weight {req.weight}')
                if self.running_round:
                    self._debug(f'got NEW_ROUND from "{token}" but round is already running')
                self.agent_new_round[token] = True
                self.agent_weight[token] = req.weight
            elif isinstance(req, ProtoConverged):
                self._debug(f'got CONVERGED from "{token}"')
                if not self.running_round:
                    self._debug(f'got CONVERGED from "{token}" but round is not yet running')
                self.agent_converged[token] = True
            elif isinstance(req, ProtoNotConverged):
                self._debug(f'got NOT_CONVERGED from "{token}"')
                if not self.running_round:
                    self._debug(f'got NOT_CONVERGED from "{token}" but round is not yet running')
                self.agent_converged[token] = False
            else:
                self._debug(f'got unexpected request from "{token}": {req!r}')

            if not self.running_round and all(self.agent_new_round.values()):
                self._debug('===== STARTING A NEW ROUND =====')
                self.round_counter += 1
                self.running_round = True
                self.agent_new_round = { token: False for token in self.tokens }
                self.agent_converged = { token: False for token in self.tokens }
                mean_weight = sum(self.agent_weight.values()) / len(self.tokens)
                for token, psocket in self.agent_psockets.items():
                    await psocket.send(ProtoNewRoundNotification(self.round_counter, mean_weight))

            self._debug(f"checking DONE: {sum(list(map(int, list(self.agent_converged.values()))))}/{len(self.tokens)} converged")
            if self.running_round and all(self.agent_converged.values()):
                self._debug('===== ALL NODES CONVERGED! DONE =====')
                self.running_round = False
                for token, psocket in self.agent_psockets.items():
                    await psocket.send(ProtoDone(self.round_counter))