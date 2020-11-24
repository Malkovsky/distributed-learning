from utils.consensus_node import ConsensusNode
import sys
import time
from itertools import cycle
from tqdm.notebook import tqdm
import torch
import os
import numpy as np
import pickle5 as pickle
import wide_resnet_submodule.config as cf


class MasterNode:
    def __init__(self,
                 node_names,
                 weights: dict,
                 train_loaders: dict,
                 test_loader,
                 fit_step,
                 update_params,
                 lr=0.02,
                 w_schedule=None,
                 epoch=200,
                 epoch_len=391,
                 update_params_epoch_start=0,
                 update_params_period=1,
                 use_cuda=False,
                 resume_path=None,
                 save_path='./checkpoint/',
                 session=None,
                 verbose=1):
        """
        Class implementing master node in consensus network.
        :param node_names: list of string with node's names in network
        :param weights: topology of network graph
        :param train_loaders: dict of train loaders, each must be iterable
        :param test_loader:
        :param fit_step: function witch train node.model on one part of data which take from node.train_loader.
        :param update_params: function witch update node.model.parameters using node.weights based on node.neighbors.
        :param lr: gradient learning rate
        :param w_schedule: schedule of weights update. None/decrease/increase
        :param epoch: number of epoch
        :param epoch_len: number of batches in each epoch
        :param update_params_epoch_start: the first epoch from which consensus begins
        :param update_params_period: consensus iteration period
        :param use_cuda: set True to use CUDA
        :param resume_path: path to saved models
        :param session: unique session identifier
        :param verbose: verbose mode
        """
        self.node_names = node_names

        self.model = None
        self.model_args = None
        self.model_kwargs = None
        self.optimizer = None
        self.opt_args = None
        self.opt_kwargs = None
        self.error = None
        self.error_args = None
        self.error_kwargs = None

        self.lr = lr
        self.w_schedule = w_schedule

        self.fit_step = fit_step
        self.update_params = update_params

        self.stat_funcs = None
        self.statistics = None

        self.weights: dict = weights
        self.network = None

        self.train_loaders = train_loaders
        self.test_loader = test_loader

        self.epoch: int = epoch
        self.epoch_len: int = epoch_len
        self.update_params_epoch_start: int = update_params_epoch_start
        self.update_params_period: int = update_params_period

        self.use_cuda = use_cuda
        self.resume_path = resume_path
        self.save_path = save_path
        self.session = session

        self.verbose = verbose
        self.debug_file = sys.stdout

    def _print_debug(self, msg, verbose):
        """
        Print msg if valid verbose mode
        :param msg: string of message
        :param verbose: verbose mode
        :return: self
        """
        if verbose <= self.verbose:
            print(msg, file=self.debug_file)
        return self

    def set_model(self, model, *args, **kwargs):
        """
        Sets self.model on given model
        :param model: some model with interface like models in pytorch
        :param args: other unnamed params
        :param kwargs: other named params
        :return: self
        """
        self.model = model
        self.model_args = args
        self.model_kwargs = kwargs
        self._print_debug(f"Master set model={self.model}, args={args}, kwargs={kwargs}", 3)
        return self

    def set_optimizer(self, optimizer, *args, **kwargs):
        """
        Sets self optimizer on given optimizer
        :param optimizer: some pytorch optimizer
        :param args: other unnamed params
        :param kwargs: other named params
        :return: self
        """
        self.optimizer = optimizer
        self.opt_args = args
        self.opt_kwargs = kwargs
        self._print_debug(f"Master set optimizer={self.optimizer}, args={args}, kwargs={kwargs}", 3)
        return self

    def set_error(self, error, *args, **kwargs):
        """
        Sets self error function on given func
        :param error: some error function
        :param args: other unnamed params
        :param kwargs: other named params
        :return: self
        """
        self.error = error
        self.error_args = args
        self.error_kwargs = kwargs
        self._print_debug(f"Master set error={self.error}, args={args}, kwargs={kwargs}", 3)
        return self

    def set_stats(self, stat_funcs=None, statistics=None, *args, **kwargs):
        """
        Sets self stat_funcs and self statistics on stat_funcs and statistics or loads it from self.resume_path
        :param stat_funcs: dict of statistic functions
        :param statistics: dict for saving statistics for each nodes: statistics['func_name']['node_name']['values/epoch/tmp']
        :param args: ther unnamed params
        :param kwargs: other named params
        :return: self
        """
        if self.resume_path:
            path = self.resume_path + os.sep

            with open(path + 'stat_funcs.pickle', 'rb') as f:
                self.stat_funcs = pickle.load(f)

            with open(path + 'statistics.pickle', 'rb') as f:
                self.statistics = pickle.load(f)

            self._print_debug(f'Master successfully loads stat_funcs={self.stat_funcs.keys()} from {self.resume_path}',
                              verbose=3)
            self._print_debug(f'Master successfully loads statistics={self.statistics.keys()} from {self.resume_path}',
                              verbose=3)

        elif stat_funcs and statistics:
            self.stat_funcs = stat_funcs
            self.statistics = statistics
            self._print_debug(f'Master set stat_funcs={self.stat_funcs.keys()}', verbose=3)
            self._print_debug(f'Master set statistics={self.statistics.keys()}', verbose=3)
        else:
            self._print_debug('Error! stat_funcs or statistics not found!', verbose=0)
            exit(-1)

        return self

    def save_stats(self, *args, **kwargs):
        """
        Saves all statistics to self.save_path/self.session/
        :return: self
        """
        if not self.session:
            return self
        self._print_debug('Saving stats...', verbose=1)
        path = self.save_path + self.session + os.sep
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + 'stat_funcs.pickle', 'wb') as f:
            pickle.dump(self.stat_funcs, f)
        with open(path + 'statistics.pickle', 'wb') as f:
            pickle.dump(self.statistics, f)
        return self

    def save_models(self, epoch: int, *args, **kwargs):
        """
        Saves model params
        :param epoch: epoch number
        :param args: other unnamed params
        :param kwargs: other named params
        :return: self
        """
        if not self.session:
            return self
        self._print_debug('Saving models...', verbose=1)
        path = self.save_path + self.session + os.sep
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + 'epoch.pickle', 'wb') as f:
            pickle.dump(epoch, f)

        for node_name, node in self.network.items():
            state = {
                'model': node.model.module if self.use_cuda else node.model,
            }
            torch.save(state, path + node_name + '.t7')
        return self

    def equalize_all_model_params(self, node_name=None):
        if not self.network:
            self._print_debug(f"Error! Network does not exist!", verbose=0)
            exit(-1)
        if not node_name:
            node_name = self.network.keys()[0]
        if node_name not in self.network:
            self._print_debug(f"Error! {node_name} does not exist!", verbose=0)
            exit(-1)
        model = self.network[node_name].model
        if not model:
            self._print_debug(f"Error! {node_name} has no model!", verbose=0)
            exit(-1)
        for name, node in self.network.items():
            if name != node_name:
                for p, pp in zip(node.model.parameters(), model.parameters()):
                    p.data = 1 * pp.data
        return self

    def initialize_nodes(self):
        """
        Initialize consensus nodes based on available information
        :return: self
        """
        self.network = {name: ConsensusNode(name=name,
                                            lr=self.lr,
                                            weights=self.weights[name],
                                            train_loader=cycle(iter(self.train_loaders[name])),
                                            use_cuda=self.use_cuda,
                                            verbose=self.verbose)
                        for name in self.node_names}

        for node_name, node in self.network.items():
            if self.resume_path:
                path = self.resume_path + os.sep + node_name + '.t7'
                node.set_model(self.model, resume_path=path)
            elif self.model:
                node.set_model(self.model, *self.model_args, **self.model_kwargs)
            if self.optimizer:
                node.set_optimizer(self.optimizer, *self.opt_args, **self.opt_kwargs)
            if self.error:
                node.set_error(self.error, *self.error_args, **self.error_kwargs)
            node.set_neighbors({neighbor_name: self.network[neighbor_name]
                                for neighbor_name in self.weights[node_name]
                                if neighbor_name != node_name})
        return self

    def start_consensus(self):
        """
        Main train function
        :return: self
        """
        self._print_debug(f'Master started\n', verbose=0)
        start_time = time.time()
        start_epoch = 1
        if self.resume_path:
            path = self.resume_path + os.sep + 'epoch.pickle'
            with open(path, 'rb') as f:
                start_epoch = pickle.load(f) + 1

        for epoch in tqdm(range(start_epoch, start_epoch + self.epoch)):
            start_epoch_time = time.time()
            self.do_epoch(epoch)
            self.save_stats()
            self.save_models(epoch=epoch)
            epoch_time = time.time() - start_epoch_time
            self._print_debug(f'Epoch {epoch} ended in: %d:%02d:%02d\n' % (cf.get_hms(epoch_time)),
                              verbose=1)
        elapsed_time = time.time() - start_time
        self._print_debug('Master ended in : %d:%02d:%02d\n' % (cf.get_hms(elapsed_time)), verbose=0)
        return self

    def do_epoch(self, epoch):
        """
        Train all network one epoch
        :param epoch: epoch number
        :return: self
        """
        self._print_debug(f"Epoch {epoch}:", verbose=1)
        for it in range(1, self.epoch_len + 1):
            global_iter = (epoch - 1) * self.epoch_len + it

            # training each model one step (batch for ex.)
            for node_name, node in self.network.items():
                self.fit_step(self, node, epoch, use_cuda=self.use_cuda)

            # Consensus starting from the {self.update_params_epoch_start}th epoch
            # with a period of {self.update_params_period}
            if epoch >= self.update_params_epoch_start \
                    and global_iter % self.update_params_period == 0:
                for node_name, node in self.network.items():
                    node.ask_params()

                for node_name, node in self.network.items():
                    self.update_params(node, epoch, global_iter, w_schedule=self.w_schedule)

        # Save stat each epoch
        for node_name, node in self.network.items():
            for func_name, func in self.stat_funcs.items():
                value = func(master_node=self, node=node, epoch=epoch,
                             use_cuda=self.use_cuda)
                self.statistics[func_name][node_name]['values'].append(value)
                self.statistics[func_name][node_name]['epoch'].append(epoch)
                if func_name != 'param_dev':
                    self._print_debug(f"Node {node_name}: epoch {epoch},"
                                      f" {func_name}= {value:.2f}", verbose=2)

        # Calculate parameter deviation
        if 'param_dev' in self.stat_funcs:
            average_params = sum([self.statistics['param_dev'][node_name]['values'][-1]
                                  for node_name in self.network]) / len(self.network)
            for node_name, node in self.network.items():
                value = np.linalg.norm(self.statistics['param_dev'][node_name]['values'][-1] - average_params)
                self.statistics['param_dev'][node_name]['values'][-1] = value
                self._print_debug(f"Node {node_name}: epoch {epoch},"
                                  f" param_dev= {value:.2f}", verbose=2)

        return self
