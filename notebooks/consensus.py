'''
## Инфраструктура для консенсуса

Как происходит коммуникация: каждый агент получает на каждого соседа пару (read, write) asyncio очередей для общения.
Аналогично с мастер-нодой консенсуса.

Протокол консенсуса в данный момент такой:
1. Мастер-нода получает топологию и агентов;
2. Мастер-нода устанавливает связь между собой и агентами (передает сокеты);
3. Мастер-нода передает сокеты для общения между агентами;
4. Мастер передает всем агентам NETWORK_READY;
5. Агенты заявляют о готовности начать новый раунд консенсуса (отправляют NEW_ROUND мастеру);
6. Когда все агенты готовы, мастер посылает всем NEW_ROUND --- начинается новый раунд консенсуса;
7. Агенты вычисляют функцию;
8. Агенты запрашивают у соседей значение их функции (отправка REQUEST_VALUE) и получают от соседей посчитанное значение;
9. Когда агент понимает, что он сошелся с соседями, он посылает мастеру CONVERGED (но при этом агент все еще должен участвовать в коммуникации с соседями);
10. Если агент понял, что сходимость не достигнута, то он может отменить CONVERGED отправкой NOT_CONVERGED;
11. Когда все агенты заявили о том, что значение сошлось, мастер посылает всем DONE;
12. jmp 5
'''

import numpy as np
from threading import Thread, Lock, RLock
import asyncio, sys

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
        self.lock = RLock()
        self.shutdown_q = shutdown_q
        
        self.running_round = False
        self.agent_new_round = dict()
        self.agent_converged = dict()
        
        self.debug = debug
        
    def _debug(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)
    
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
        eps = 0.99 / np.max(outdeg) # eps \in (0, 1/max deg)
        P = np.eye(outdeg.shape[0]) - eps * L
        print('Perron matrix:\n{}'.format(P))
        P_eig = np.linalg.eigvals(P)
        P_eig.sort()
        print('Eigenvalues: {}'.format(P_eig))
        print('Convergence speed: {}'.format(P_eig[1]**2))
        
    def register_agent(self, agent):
        with self.lock:
            if agent.token not in self.tokens:
                raise IllegalArgumentException('Agent with token {} is not presented in given topology'.format(agent.token))
            self.agents[agent.token] = agent
            self._debug('Got {}/{} agents'.format(len(self.agents.keys()), len(self.tokens)), file=sys.stderr)
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
            
        for token, (r, w) in self.agents_sockets.items():
            asyncio.create_task(w.put(NETWORK_READY), name='master put NETWORK_READY')
            
    async def serve(self):
        self._debug('Master: serving...', file=sys.stderr)
        self.agent_new_round = { token: False for token in self.tokens }
        self.agent_converged = { token: False for token in self.tokens }
        while True:
            monitor_shutdown = asyncio.create_task(self.shutdown_q.get(), name='master check shutdown')
            monitor_agents = { token: asyncio.create_task(r.get(), name='master check agent "{}"'.format(token)) 
                              for token, (r, w) in self.agents_sockets.items() }
            done, pending = await asyncio.wait({monitor_shutdown}.union(set(monitor_agents.values())), return_when=asyncio.FIRST_COMPLETED)
            if monitor_shutdown in done:
                self._debug('Master: ===== SHUTDOWN =====', file=sys.stderr)
                for token, (r, w) in self.agents_sockets.items():
                    await w.put(SHUTDOWN)
                self.shutdown_q.task_done()
                break
            
            check_new_round = False
            check_done = False
            
            for token, monitor in monitor_agents.items():
                if monitor in done:
                    req = monitor.result()
                    if req == NEW_ROUND:
                        self._debug('Master: got NEW_ROUND from "{}"'.format(token), file=sys.stderr)
                        if self.running_round:
                            self._debug('Master: got NEW_ROUND from "{}" but round is already running'.format(token), file=sys.stderr)
                        with self.lock:
                            self.agent_new_round[token] = True
                            check_new_round = True
                    elif req == CONVERGED:
                        self._debug('Master: got CONVERGED from "{}"'.format(token), file=sys.stderr)
                        if not self.running_round:
                            self._debug('Master: got CONVERGED from "{}" but round is not yet running'.format(token), file=sys.stderr)
                        with self.lock:
                            self.agent_converged[token] = True
                            check_done = True
                    elif req == NOT_CONVERGED:
                        self._debug('Master: got NOT_CONVERGED from "{}"'.format(token), file=sys.stderr)
                        if not self.running_round:
                            self._debug('Master: got NOT_CONVERGED from "{}" but round is not yet running'.format(token), file=sys.stderr)
                        with self.lock:
                            self.agent_converged[token] = False
                    else:
                        self._debug('Master: got unexpected request from "{}": {}'.format(token, req), file=sys.stderr)
                    
                    r, _ = self.agents_sockets[token]
                    r.task_done()
                    
            for t in pending:
                t.cancel()
                
            with self.lock:
                if not self.running_round and all(self.agent_new_round.values()): 
                    self._debug('Master: ===== STARTING A NEW ROUND =====', file=sys.stderr)
                    self.running_round = True
                    self.agent_new_round = { token: False for token in self.tokens }
                    self.agent_converged = { token: False for token in self.tokens }
                    for token, (r, w) in self.agents_sockets.items():
                        await w.put(NEW_ROUND)

                self._debug("Master: checking DONE: {}/{} converged".format( sum(list(map(int, list(self.agent_converged.values())))) , len(self.tokens) ), file=sys.stderr)
                if self.running_round and all(self.agent_converged.values()):
                    self._debug('Master: ===== ALL NODES CONVERGED! DONE =====', file=sys.stderr)
                    self.running_round = False
                    for token, (r, w) in self.agents_sockets.items():
                        await w.put(DONE)
                
                
class ConsensusAgent:
    def __init__(self, token, debug=False, eps=1e-10):
        self.token = token
        self.neighbor_sockets = dict()
        self.master_sockets = None
        
        self.network_ready = False
        
        self.eps = eps
        self.debug = debug
        
    def _debug(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)
        
    def set_master(self, master_sockets):
        self.master_sockets = master_sockets
        self._debug('Agent "{}" heard from master'.format(self.token), file=sys.stderr)
    
    def set_neighbors(self, neighbor_sockets: dict):
        self.neighbor_sockets = neighbor_sockets
        self._debug('Agent "{}" got neighbors from master'.format(self.token), file=sys.stderr)
    
    async def run_round(self, value, weight):
        self._debug('Agent "{}": running new round with v={}, w={}'.format(self.token, value, weight), file=sys.stderr)
        if not self.network_ready:
            self._debug('Agent "{}" initialized. Waiting for NETWORK_READY'.format(self.token), file=sys.stderr)
            rdy = await self.master_sockets[0].get()
            self._debug('Agent "{}" got {}'.format(self.token, rdy), file=sys.stderr)
            self.master_sockets[0].task_done()
            self.network_ready = rdy == NETWORK_READY
            if not self.network_ready:
                return rdy
        
        # clear queues in case DONE was transmitted in the middle of the previous round
        self._debug('Agent "{}" clearing queues'.format(self.token), file=sys.stderr)
        for token, (r, w) in self.neighbor_sockets.items():
            while r.qsize() > 0:
                r.get_nowait()
                r.task_done()
        
        self._debug('Agent "{}" sending NEW_ROUND to master'.format(self.token), file=sys.stderr)
        await self.master_sockets[1].put(NEW_ROUND)
        new_round = await self.master_sockets[0].get()
        self.master_sockets[0].task_done()
        if new_round != NEW_ROUND:
            return new_round
        
        self._debug('Agent "{}": NEW_ROUND ack!'.format(self.token), file=sys.stderr)
        
        converged_flag_set = False
        done_flag = False
        x = value
        neighbor_count = len(self.neighbor_sockets.keys())

        while not done_flag:
            self._debug('Agent "{}": requesting values from neighbors'.format(self.token), file=sys.stderr)
            for token, (r, w) in self.neighbor_sockets.items():
                await w.put(REQUEST_VALUE)

            neighbor_values_weights = {}

            while len(neighbor_values_weights.keys()) != neighbor_count:
                if self.master_sockets[0].qsize() > 0:
                    res = self.master_sockets[0].get_nowait()
                    self.master_sockets[0].task_done()
                    if res == DONE:
                        self._debug('Agent "{}": got DONE from master!!!'.format(self.token), file=sys.stderr)
                        done_flag = True
                        break
                    elif res == SHUTDOWN:
                        return SHUTDOWN # TODO: maybe it is better to throw an exception
                    else:
                        self._debug('Unexpected request from master: {}'.format(res), file=sys.stderr)
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
                    self.master_sockets[0].task_done()
                    if res == DONE:
                        self._debug('Agent "{}": got DONE from master!!!'.format(self.token), file=sys.stderr)
                        done_flag = True
                        break
                    elif res == SHUTDOWN:
                        return SHUTDOWN # TODO: maybe it is better to throw an exception
                    else:
                        self._debug('Unexpected request from master: {}'.format(res), file=sys.stderr)
                for token, task in monitor_neighbors.items():
                    if task in done:
                        res = task.result()
                        r, w = self.neighbor_sockets[token]
                        r.task_done()
                        if isinstance(res, str):
                            if res == REQUEST_VALUE:
                                self._debug('Agent "{}": sending values to "{}"'.format(self.token, token), file=sys.stderr)
                                await w.put((x, weight))
                            else:
                                self._debug('Agent "{}" got unexpected request from "{}": {}'.format(self.token, token, res), file=sys.stderr)
                        else: # consider it a (value, weight)
                            self._debug('Agent "{}": got values from "{}"!'.format(self.token, token), file=sys.stderr)
                            neighbor_values_weights[token] = res
                            
            if done_flag: break

            neighbor_values_weights[self.token] = (x, weight)
            
            x = np.sum([v * w for (v, w) in neighbor_values_weights.values()], axis=0) / \
                np.sum([w for (v, w) in neighbor_values_weights.values()])
            
            c = np.all([ np.isclose(x, v, rtol=self.eps) for (v, w) in neighbor_values_weights.values()])
            
            self._debug('Agent "{}": updated value = {}, c={}'.format(self.token, x, c), file=sys.stderr)
            
            if c:
                if not converged_flag_set:
                    self._debug('Agent "{}": sending CONVERGED to master'.format(self.token), file=sys.stderr)
                    await self.master_sockets[1].put(CONVERGED)
                    converged_flag_set = True
            else:
                if converged_flag_set:
                    self._debug('Agent "{}": sending NOT_CONVERGED to master'.format(self.token), file=sys.stderr)
                    await self.master_sockets[1].put(NOT_CONVERGED)
                    converged_flag_set = False
            
        self._debug('Agent "{}": final result: {}'.format(self.token, x), file=sys.stderr)
        return x
        