import numpy as np
import asyncio
import sys
from typing import Dict, List

from .pickled_socket import PickledSocketWrapper
from .protocol import *
from .psocket_selector import PSocketSelector


class ConsensusAgent:
    MASTER_TOKEN = 'MASTER'

    def __init__(self, token, host, port, master_host, master_port, convergence_eps=1e-4, public_host_port=None, debug=False):
        self.token = token
        self.host = host
        self.port = port
        self.master_host = master_host
        self.master_port = master_port
        self.public_host_port = (host, port) if public_host_port is None else public_host_port

        self.neighbor_count = None
        self.neighbor_psockets_incoming: Dict[Any, PickledSocketWrapper] = dict()
        self.neighbor_incoming_psocket_selector = PSocketSelector()
        self.neighbor_psockets_outcoming: Dict[Any, PickledSocketWrapper] = dict()
        self.neighbor_outcoming_psocket_selector = PSocketSelector()
        self.master_psocket = None

        self.network_ready = False
        self.neighbor_weights = None
        self.convergence_rate = None
        self.convergence_eps = convergence_eps
        self.current_round = None
        self.value_history = {}
        self.value_history_request_counts = {}
        self.run_once_value = None
        self.run_once_request_count = 0
        self.run_once_iteration = 0

        self.debug = debug

        self.server: asyncio.Server = None

    def _debug(self, *args, **kwargs):
        if self.debug:
            if 'file' not in kwargs.keys():
                print(f'Agent "{self.token}":', *args, **kwargs, file=sys.stderr)
            else:
                print(f'Agent "{self.token}":', *args, **kwargs)

    async def serve_forever(self):
        self.server = await asyncio.start_server(self.handle_connection, self.host, self.port)
        print(f'Serving on {self.server.sockets[0].getsockname()}')
        try:
            async with self.server:
                serve_task = self.server.get_loop().create_task(self._serve())
                await self.server.serve_forever()
        except:
            serve_task.cancel()
            raise

    async def _serve(self):
        self._debug('Serving...')
        while not self.network_ready:
            await asyncio.sleep(0.25)
        while True:
            token, req = await self.neighbor_incoming_psocket_selector.recv(0.5)
            if token is None:
                await asyncio.sleep(0.25)
                continue
            if isinstance(req, ProtoRunOnceValueRequest):
                if req.iteration < self.run_once_iteration:
                    msg = f'!! got (run once) value request from agent({token}) and it is obsolete! Something is wrong!'
                    print(msg)
                    raise ProtoErrorException(msg)
                self._debug(f'Got (run once) value request from agent({token}) for iteration {req.iteration}')
                async def respond(token, req):
                    while self.run_once_value is None or self.run_once_iteration != req.iteration:  # wait for the value
                        await asyncio.sleep(0.05)
                    self._debug(f'Sending to {token} (run once) value (iteration {self.run_once_iteration})')
                    await self.neighbor_psockets_incoming[token].send(
                        ProtoRunOnceValueResponse(self.run_once_iteration, self.run_once_value)
                    )
                    self.run_once_request_count += 1
                    if self.run_once_request_count == self.neighbor_count:
                        self.run_once_value = None
                        self.run_once_request_count = 0
                asyncio.create_task(respond(token, req))
                continue

            if not isinstance(req, ProtoValueRequest):
                msg = f'From agent({token}). Expected value request, got: {req!r}'
                self._debug(msg)
                raise ProtoErrorException(msg)
            key = (req.round_id, req.round_iteration)
            self._debug(f'Got value request from agent({token}) for {key}')
            if key[0] < self.current_round: # obsolete request:
                self._debug(f'Request is obsolete, skipping it')
                continue
            async def respond(token, req):
                key = (req.round_id, req.round_iteration)
                while key not in self.value_history.keys() and key[0] >= self.current_round: # wait until it is calculated
                    await asyncio.sleep(0.05)
                val = self.value_history[key]
                self._debug(f'Sending to {token} value for {req.round_id, req.round_iteration}')
                await self.neighbor_psockets_incoming[token].send(ProtoValueResponse(req.round_id, req.round_iteration, val))
                if not key in self.value_history_request_counts.keys():
                    self.value_history_request_counts[key] = 0
                self.value_history_request_counts[(req.round_id, req.round_iteration)] += 1
                if self.value_history_request_counts[key] == self.neighbor_count:
                    self._debug(f'All neighbors got our value for {key}! Deleting it')
                    del self.value_history_request_counts[key]
                    del self.value_history[key]
            asyncio.create_task(respond(token, req))
            if self.current_round is not None: # delete obsolete values
                to_del = []
                for k in self.value_history.keys():
                    if k[0] < self.current_round:
                        to_del.append(k)
                for k in to_del:
                    self._debug(f'Key {k} is obsolete: current round is {self.current_round}. Deleting it')
                    del self.value_history[k]
                    if k in self.value_history_request_counts.keys():
                        del self.value_history_request_counts[k]

    async def handle_connection(self, reader, writer):  # these connections are from other agents
        psocket = PickledSocketWrapper(reader, writer)
        data = await psocket.recv()
        remote_name = psocket.writer.get_extra_info("peername")
        if not isinstance(data, ProtoRegister):
            msg = f'From: {remote_name}. Expected registration data, got: {data!r}'
            self._debug(msg)
            raise ProtoErrorException(msg)
        await psocket.send(ProtoOk())
        self.neighbor_psockets_incoming[data.token] = psocket
        self._debug(f'From: {remote_name}. Registered {data.token}!')
        self.neighbor_incoming_psocket_selector.add(data.token, psocket)

    async def do_handshake(self):
        self._debug('Performing handshake with master')
        reader, writer = await asyncio.open_connection(self.master_host, self.master_port)
        self.master_psocket = PickledSocketWrapper(reader, writer)
        await self.master_psocket.send(ProtoRegister(self.token, self.public_host_port[0], self.public_host_port[1]))
        ok = await self.master_psocket.recv()
        remote_name = self.master_psocket.writer.get_extra_info("peername")
        if not isinstance(ok, ProtoOk):
            msg = f'From master({remote_name}). Expected OK, got: {ok!r}'
            self._debug(msg)
            raise ProtoErrorException(msg)
        self._debug('Successfully registered. Waiting for neighborhood data...')
        neighborhood_data = await self.master_psocket.recv()
        if not isinstance(neighborhood_data, ProtoNeighborhoodData):
            msg = f'From master({remote_name}). Expected neighborhood data, got: {neighborhood_data!r}'
            self._debug(msg)
            raise ProtoErrorException(msg)
        await self.master_psocket.send(ProtoOk())
        self.neighbor_count = len(neighborhood_data.neighbors)
        self._debug('Got neighborhood data. Waiting for consensus epsilon...')
        neighbor_weights_data = await self.master_psocket.recv()
        if not isinstance(neighbor_weights_data, ProtoNeighborWeights):
            msg = f'From master({remote_name}). Expected neighbor weights data, got: {neighbor_weights_data!r}'
            self._debug(msg)
            raise ProtoErrorException(msg)
        self.neighbor_weights = neighbor_weights_data.weights
        self.convergence_rate = neighbor_weights_data.convergence_rate
        await self.master_psocket.send(ProtoOk())
        self._debug('Got consensus epsilon!')

        self._debug('Performing handshake with neighbors...')
        for (n_token, n_host, n_port) in neighborhood_data.neighbors:
            self._debug(f'Performing handshake with {n_token}')
            n_reader, n_writer = await asyncio.open_connection(n_host, n_port)
            n_psocket = PickledSocketWrapper(n_reader, n_writer)
            self.neighbor_psockets_outcoming[n_token] = n_psocket

            await n_psocket.send(ProtoRegister(self.token, self.host, self.port))
            ok = await n_psocket.recv()
            if not isinstance(ok, ProtoOk):
                remote_name = self.master_psocket.writer.get_extra_info("peername")
                msg = f'From agent({remote_name}). Expected Ok data, got: {ok!r}'
                self._debug(msg)
                raise ProtoErrorException(msg)
            self._debug(f'Success!')
            self.neighbor_outcoming_psocket_selector.add(n_token, n_psocket)

        self._debug(f'We are ready for consensus now!')
        self.network_ready = True

    async def run_round(self, value, weight, convergence_eps=None):
        if not self.network_ready:
            await self.do_handshake()

        self._debug(f'running new round with v={value}, w={weight}')
        self._debug('sending new round request to master')
        await self.master_psocket.send(ProtoNewRoundRequest(weight))

        self._debug(f'waiting for new round notification from master')
        resp = await self.master_psocket.recv()
        if not isinstance(resp, ProtoNewRoundNotification):
            msg = f'From master. Expected new round notification, got: {resp!r}'
            self._debug(msg)
            raise ProtoErrorException(msg)
        self.current_round = resp.round_id
        current_round_iteration = 0
        mean_weight = resp.mean_weight
        self._debug(f'Got new round notification from master! Round id: {resp.round_id}')

        if convergence_eps is None:
            convergence_eps = self.convergence_eps
        converged_flag_set = False
        done_flag = False
        shutdown_flag = False
        y = value * (weight / mean_weight)
        self._debug(f'Adding value to history with key {(self.current_round, current_round_iteration)}')
        self.value_history[(self.current_round, current_round_iteration)] = y.copy()
        self.value_history_request_counts[(self.current_round, current_round_iteration)] = 0

        while not done_flag and not shutdown_flag:
            for token, psocket in self.neighbor_psockets_outcoming.items():
                self._debug(f'requesting value from agent({token}) - {(self.current_round, current_round_iteration)}')
                await psocket.send(ProtoValueRequest(self.current_round, current_round_iteration))

            neighbor_values = {}

            self.neighbor_outcoming_psocket_selector.add(self.MASTER_TOKEN, self.master_psocket)

            while len(neighbor_values.keys()) != self.neighbor_count and not done_flag and not shutdown_flag:
                # wait for values / done / shutdown
                token, req = await self.neighbor_outcoming_psocket_selector.recv()
                self._debug(f'{self.token}<-{token} <-----> {req!r}')
                if token == self.MASTER_TOKEN:
                    data = req
                    if isinstance(data, ProtoDone):
                        self._debug('got DONE from master!!!')
                        done_flag = True
                    elif isinstance(data, ProtoShutdown):
                        self._debug('Got SHUTDOWN from master')
                        shutdown_flag = True
                    else:
                        msg = f'Unexpected request from master: {data!r}'
                        self._debug(msg)
                        raise ProtoErrorException(msg)
                else: # it is neighbor's token
                    n_token, n_resp = token, req
                    if not isinstance(n_resp, ProtoValueResponse):
                        msg = f'From agent({n_token}). Expected value response, got: {n_resp!r}'
                        self._debug(msg)
                        raise ProtoErrorException(msg)
                    if n_resp.round_id != self.current_round or n_resp.round_iteration != current_round_iteration:
                        self._debug(
                            f'! got response from "{n_token}" from previous round/iteration:'
                            f' {(n_resp.round_id, n_resp.round_iteration)}')
                        continue  # just listen for next response
                    self._debug(f'got value from "{n_token}" for {n_resp.round_id, n_resp.round_iteration}!')
                    neighbor_values[n_token] = n_resp.value

            self.neighbor_outcoming_psocket_selector.remove(self.MASTER_TOKEN)

            if shutdown_flag:
                self.server.close()
                raise ProtoErrorException(ProtoShutdown())

            if done_flag: break

            # we've got all the values from neighbors, now we can update our value
            sum_neighbor_weights = np.sum([w for w in self.neighbor_weights.values()])
            y = (1 - sum_neighbor_weights) * y + np.sum([w * neighbor_values[n] for (n, w) in self.neighbor_weights.items()], axis=0)
            c = np.all([np.abs(y - v) <= convergence_eps for v in neighbor_values.values()])

            self._debug(f'finished iteration #{current_round_iteration}')
            current_round_iteration += 1
            self._debug(f'Adding value to history with key {(self.current_round, current_round_iteration)}')
            self.value_history[(self.current_round, current_round_iteration)] = y.copy()
            self.value_history_request_counts[(self.current_round, current_round_iteration)] = 0

            if c:
                if not converged_flag_set:
                    self._debug('sending CONVERGED to master')
                    await self.master_psocket.send(ProtoConverged())
                    converged_flag_set = True
            else:
                if converged_flag_set:
                    self._debug('sending NOT_CONVERGED to master')
                    await self.master_psocket.send(ProtoNotConverged())
                    converged_flag_set = False
        self._debug(f'final result: {y}')
        return y

    async def run_once(self, value):
        if not self.network_ready:
            await self.do_handshake()

        while self.run_once_value is not None:  # wait until all neighbors got our previous value
            await asyncio.sleep(0.02)

        self.run_once_value = value
        self.run_once_iteration += 1

        for token, psocket in self.neighbor_psockets_outcoming.items():
            self._debug(f'requesting (run once) value from agent({token})')
            await psocket.send(ProtoRunOnceValueRequest(self.run_once_iteration))

        neighbor_values = {}

        self.neighbor_outcoming_psocket_selector.add(self.MASTER_TOKEN, self.master_psocket)

        while len(neighbor_values.keys()) != self.neighbor_count:
            # wait for values / done / shutdown
            token, req = await self.neighbor_outcoming_psocket_selector.recv()
            if token == self.MASTER_TOKEN:
                data = req
                if isinstance(data, ProtoShutdown):
                    self._debug('Got SHUTDOWN from master')
                    self.server.close()
                    raise ProtoErrorException(ProtoShutdown())
                else:
                    msg = f'Unexpected request from master: {data!r}'
                    self._debug(msg)
                    raise ProtoErrorException(msg)
            else: # it is neighbor's token
                n_token, n_resp = token, req
                if isinstance(n_resp, ProtoValueResponse):
                    self._debug(
                        f'! got response from "{n_token}" from previous round/iteration:'
                        f' {(n_resp.round_id, n_resp.round_iteration)}')
                    continue  # skip
                if not isinstance(n_resp, ProtoRunOnceValueResponse) or n_resp.iteration != self.run_once_iteration:
                    msg = f'From agent({n_token}). Expected (run once) value response for iteration {self.run_once_iteration}, got: {n_resp!r}'
                    self._debug(msg)
                    raise ProtoErrorException(msg)
                self._debug(f'got (run once) value from "{n_token}"!')
                neighbor_values[n_token] = n_resp.value

        self.neighbor_outcoming_psocket_selector.remove(self.MASTER_TOKEN)

        # we've got all the values from neighbors, now we can update our value
        sum_neighbor_weights = np.sum([w for w in self.neighbor_weights.values()])
        value = (1 - sum_neighbor_weights) * value + np.sum([w * neighbor_values[n] for (n, w) in self.neighbor_weights.items()], axis=0)
        self._debug(f'final result: {value}')
        return value
