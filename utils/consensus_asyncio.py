'''
## Инфраструктура для консенсуса

Как происходит коммуникация: каждый агент получает на каждого соседа пару (read, write) asyncio очередей для общения.
Аналогично с мастер-нодой консенсуса.

Протокол консенсуса в данный момент такой:
1. Мастер-нода получает топологию и агентов;
2. Мастер-нода устанавливает связь между собой и агентами (передает сокеты);
3. Мастер-нода передает сокеты для общения между агентами;
4. Мастер передает всем агентам NETWORK_READY;
5. Агенты заявляют о готовности начать новый раунд консенсуса (отправляют (NEW_ROUND, _свой вес_) мастеру);
6. Когда все агенты готовы, мастер посылает всем (NEW_ROUND, _средний вес_) --- начинается новый раунд консенсуса;
7. Агенты вычисляют функцию;
8. Агенты запрашивают у соседей значение их функции (отправка REQUEST_VALUE) и получают от соседей посчитанное значение;
9. Когда агент понимает, что он сошелся с соседями, он посылает мастеру CONVERGED (но при этом агент все еще должен участвовать в коммуникации с соседями);
10. Если агент понял, что сходимость не достигнута, то он может отменить CONVERGED отправкой NOT_CONVERGED;
11. Когда все агенты заявили о том, что значение сошлось, мастер посылает всем DONE;
12. jmp 5
'''

import numpy as np
import asyncio, sys
import functools

# asyncio.get_event_loop().set_debug(True)

NEW_ROUND = 'NEW_ROUND'
REQUEST_VALUE = 'REQUEST_VALUE'
CONVERGED = 'CONVERGED'
NOT_CONVERGED = 'NOT CONVERGED'
DONE = 'DONE'
NETWORK_READY = 'NETWORK_READY'
SHUTDOWN = 'SHUTDOWN'


class ConsensusNetwork:
    def __init__(self, topology, shutdown_q, debug=False):
        self.topology = topology
        self.tokens = list(set(np.array(topology).flatten()))
        self.agents = dict()
        self.agents_sockets = dict()
        self.shutdown_q = shutdown_q
        
        self.running_round = False
        self.agent_new_round = dict()
        self.agent_weight = dict()
        self.agent_converged = dict()
        
        self.debug = debug
        
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
        P = np.eye(outdeg.shape[0]) - self.__calc_eps() * L
        print('Perron matrix:\n{}'.format(P))
        P_eig = np.linalg.eigvals(P)
        P_eig.sort()
        print('Eigenvalues: {}'.format(P_eig))
        print('Convergence speed: {}'.format(np.abs(P_eig[1])))
        
    @functools.lru_cache
    def __calc_eps(self):
        E = np.array([ [
                int((u, v) in self.topology or (v, u) in self.topology)
                for v in self.tokens
            ] for u in self.tokens])
        outdeg = np.sum(E, axis=1)
        eps = 0.95 / np.max(outdeg) # eps \in (0, 1/max deg)
        return eps
        
    def register_agent(self, agent):
        if agent.token not in self.tokens:
            raise IllegalArgumentException('Agent with token {} is not presented in given topology'.format(agent.token))
        self.agents[agent.token] = agent
        self._debug(f'Got {len(self.agents.keys())}/{len(self.tokens)} agents')
        if len(self.agents.keys()) == len(self.tokens):
            self.initialize_agents()
                
    def initialize_agents(self):
        for token, agent in self.agents.items():
            master_r, master_w = asyncio.Queue(), asyncio.Queue()
            self.agents_sockets[token] = (master_r, master_w)
            agent.set_master((master_w, master_r))
        
        sockets = dict()
        for token, agent in self.agents.items():
            neighbors = [u if token == v else v for (u, v) in self.topology if token == u or token == v]
            data = {}
            for n_token in neighbors:
                cur_sockets = sockets.get((token, n_token), None)
                if cur_sockets is None:
                    r, w = asyncio.Queue(), asyncio.Queue()
                    sockets[(token, n_token)] = (r, w)
                    sockets[(n_token, token)] = (w, r)
                    cur_sockets = (r, w)
                data[n_token] = cur_sockets
            agent.set_neighbors(data)
            agent.set_epsilon(self.__calc_eps())
            
        for token, (r, w) in self.agents_sockets.items():
            asyncio.create_task(w.put(NETWORK_READY), name='master put NETWORK_READY')
            
    async def serve(self):
        self._debug('serving...')
        self.agent_new_round = { token: False for token in self.tokens }
        self.agent_converged = { token: False for token in self.tokens }
        while True:
            monitor_shutdown = asyncio.create_task(self.shutdown_q.get(), name='master check shutdown')
            monitor_agents = { token: asyncio.create_task(r.get(), name=f'master check agent "{token}"')
                              for token, (r, w) in self.agents_sockets.items() }
            done, pending = await asyncio.wait({monitor_shutdown}.union(set(monitor_agents.values())), return_when=asyncio.FIRST_COMPLETED)
            if monitor_shutdown in done:
                self._debug('===== SHUTDOWN =====')
                for token, (r, w) in self.agents_sockets.items():
                    await w.put(SHUTDOWN)
                break
            
            for token, monitor in monitor_agents.items():
                if monitor in done:
                    req = monitor.result()
                    if isinstance(req, tuple) and req[0] == NEW_ROUND:
                        self._debug(f'got NEW_ROUND from "{token}" with weight {req[1]}')
                        if self.running_round:
                            self._debug(f'got NEW_ROUND from "{token}" but round is already running')
                        self.agent_new_round[token] = True
                        self.agent_weight[token] = req[1]
                    elif req == CONVERGED:
                        self._debug(f'got CONVERGED from "{token}"')
                        if not self.running_round:
                            self._debug(f'got CONVERGED from "{token}" but round is not yet running')
                        self.agent_converged[token] = True
                    elif req == NOT_CONVERGED:
                        self._debug(f'got NOT_CONVERGED from "{token}"')
                        if not self.running_round:
                            self._debug(f'got NOT_CONVERGED from "{token}" but round is not yet running')
                        self.agent_converged[token] = False
                    else:
                        self._debug(f'got unexpected request from "{token}": {req}')
                    
            for t in pending:
                t.cancel()
                
            if not self.running_round and all(self.agent_new_round.values()): 
                self._debug('===== STARTING A NEW ROUND =====')
                self.running_round = True
                self.agent_new_round = { token: False for token in self.tokens }
                self.agent_converged = { token: False for token in self.tokens }
                mean_weight = sum(self.agent_weight.values()) / len(self.tokens)
                for token, (r, w) in self.agents_sockets.items():
                    await w.put((NEW_ROUND, mean_weight))

            self._debug(f"checking DONE: {sum(list(map(int, list(self.agent_converged.values()))))}/{len(self.tokens)} converged")
            if self.running_round and all(self.agent_converged.values()):
                self._debug('===== ALL NODES CONVERGED! DONE =====')
                self.running_round = False
                for token, (r, w) in self.agents_sockets.items():
                    await w.put(DONE)
                
                
class ConsensusAgent:
    def __init__(self, token, debug=False, convergence_eps=1e-4):
        self.token = token
        self.neighbor_sockets = dict()
        self.master_sockets = None
        
        self.network_ready = False
        self.consensus_eps = None
        self.convergence_eps = convergence_eps
        self.debug = debug
        
        self.round_counter = 0
        
    def _debug(self, *args, **kwargs):
        if self.debug:
            if 'file' not in kwargs.keys():
                print(f'Agent "{self.token}":', *args, **kwargs, file=sys.stderr)
            else:
                print(f'Agent "{self.token}":', *args, **kwargs)
        
    def set_master(self, master_sockets):
        self.master_sockets = master_sockets
        self._debug('heard from master')
    
    def set_neighbors(self, neighbor_sockets: dict):
        self.neighbor_sockets = neighbor_sockets
        self._debug('got neighbors from master')
        
    def set_epsilon(self, eps):
        self.consensus_eps = eps
        self._debug(f'got consensus epsilon from master: {self.consensus_eps}')
    
    async def run_round(self, value, weight):
        self.round_counter += 1
        self._debug(f'running new round with v={value}, w={weight}')
        if not self.network_ready:
            self._debug('initialized. Waiting for NETWORK_READY')
            rdy = await self.master_sockets[0].get()
            self._debug(f'got {rdy}')
            self.network_ready = rdy == NETWORK_READY
            if not self.network_ready:
                return rdy
        
        self._debug('sending NEW_ROUND to master')
        await self.master_sockets[1].put((NEW_ROUND, weight))
        resp = await self.master_sockets[0].get()
        if not isinstance(resp, tuple) or resp[0] != NEW_ROUND:
            return resp
        mean_weight = resp[1]
        
        self._debug('NEW_ROUND ack!')
        
        converged_flag_set = False
        done_flag = False
        y = value * weight / mean_weight
        neighbor_count = len(self.neighbor_sockets.keys())
        
        while not done_flag:
            self._debug('requesting values from neighbors')
            for token, (r, w) in self.neighbor_sockets.items():
                await w.put((REQUEST_VALUE, self.round_counter))

            neighbor_values = {}

            while len(neighbor_values.keys()) != neighbor_count:
                if self.master_sockets[0].qsize() > 0:
                    res = self.master_sockets[0].get_nowait()
                    if res == DONE:
                        self._debug('got DONE from master!!!')
                        done_flag = True
                        break
                    elif res == SHUTDOWN:
                        return SHUTDOWN # TODO: maybe it is better to throw an exception
                    else:
                        self._debug(f'Unexpected request from master: {res}')
                    continue
                monitor_master = asyncio.create_task(self.master_sockets[0].get())
                monitor_neighbors = { token: asyncio.create_task(r.get()) 
                                      for token, (r, w) in self.neighbor_sockets.items() }
                done, pending = await asyncio.wait({monitor_master}.union(set(monitor_neighbors.values())), 
                                             return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
                if monitor_master in done:
                    res = monitor_master.result()
                    if res == DONE:
                        self._debug('got DONE from master!!!')
                        done_flag = True
                        break
                    elif res == SHUTDOWN:
                        return SHUTDOWN # TODO: maybe it is better to throw an exception
                    else:
                        self._debug(f'Unexpected request from master: {res}')
                for token, task in monitor_neighbors.items():
                    if task in done:
                        res = task.result()
                        r, w = self.neighbor_sockets[token]
                        if not isinstance(res, tuple) or len(res) < 2:
                            self._debug(f'got unexpected request from "{token}": {res}')
                        if res[1] != self.round_counter:
                            self._debug(f'! got request/response from "{token}" from previous round: {res}')
                            continue
                        if isinstance(res[0], str) and res[0] == REQUEST_VALUE:                        
                            self._debug(f'sending values to "{token}"')
                            await w.put((y, self.round_counter))
                        else: # consider it a value
                            self._debug(f'got value from "{token}": {res}!')
                            neighbor_values[token] = res[0]
                            
            if done_flag: break

            '''
            Умеем считать 1/n \sum x_i, а хотим (\sum x_i w_i) / (\sum w_i).
            Если мы скажем мастеру свой вес, то мастер сможет вернуть нам в начале раунда средний вес агентов.
            А тогда при y_i = x_i w_i / mean_w:
            1/n \sum y_i = 1/n \sum (x_i w_i / (1/n sum w_j)) = 1/n \sum (n x_i w_i / \sum w_j) = \sum (x_i w_i) / (\sum w_j)
            '''
                
            y = y * (1 - self.consensus_eps * neighbor_count) + self.consensus_eps * np.sum(list(neighbor_values.values()), axis=0)
            
            c = np.all([ (y - v) <= self.convergence_eps for v in neighbor_values.values()])
            
            self._debug(f'updated value = {y}, c={c}')
            
            if c:
                if not converged_flag_set:
                    self._debug('sending CONVERGED to master')
                    await self.master_sockets[1].put(CONVERGED)
                    converged_flag_set = True
            else:
                if converged_flag_set:
                    self._debug('sending NOT_CONVERGED to master')
                    await self.master_sockets[1].put(NOT_CONVERGED)
                    converged_flag_set = False
        self._debug(f'final result: {y}')
        return y
        