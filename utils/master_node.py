from utils.consensus_node import ConsensusNode
import sys
import timeit
from itertools import cycle


class MasterNode:
    def __init__(self,
                 node_names,
                 weights: dict,
                 train_loaders: dict,
                 test_loader,
                 fit_step,
                 update_params,
                 calc_accuracy,
                 stat_step=50,
                 lr=0.02,
                 epoch=200,
                 epoch_len=391,
                 update_params_epoch_start=0,
                 update_params_period=1,
                 verbose=1):
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

        self.fit_step = fit_step
        self.update_params = update_params
        self.calc_accuracy = calc_accuracy

        self.weights: dict = weights
        self.network = None

        self.train_loaders = train_loaders
        self.test_loader = test_loader

        self.stat_step = stat_step
        self.epoch: int = epoch
        self.epoch_len: int = epoch_len
        self.update_params_epoch_start: int = update_params_epoch_start
        self.update_params_period: int = update_params_period

        self.verbose = verbose
        self.debug_file = sys.stdout

    def _print_debug(self, msg, verbose):
        if verbose <= self.verbose:
            print(msg, file=self.debug_file)

    def set_model(self, model, *args, **kwargs):
        self.model = model
        self.model_args = args
        self.model_kwargs = kwargs
        self._print_debug(f"Master set model={self.model}, args={args}, kwargs={kwargs}", 2)
        return self

    def set_optimizer(self, optimizer, *args, **kwargs):
        self.optimizer = optimizer
        self.opt_args = args
        self.opt_kwargs = kwargs
        self._print_debug(f"Master set optimizer={self.optimizer}, args={args}, kwargs={kwargs}", 2)
        return self

    def set_error(self, error, *args, **kwargs):
        self.error = error
        self.error_args = args
        self.error_kwargs = kwargs
        self._print_debug(f"Master set error={self.error}, args={args}, kwargs={kwargs}", 2)
        return self

    def initialize_nodes(self):
        self.network = {name: ConsensusNode(name=name,
                                            lr=self.lr,
                                            stat_step=self.stat_step,
                                            weights=self.weights[name],
                                            train_loader=cycle(iter(self.train_loaders[name])),
                                            verbose=self.verbose)
                        for name in self.node_names}

        for node_name, node in self.network.items():
            if self.model:
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
        self._print_debug(f'Master started\n', verbose=0)
        start_time = timeit.default_timer()
        for epoch in range(1, self.epoch + 1):
            start_epoch_time = timeit.default_timer()
            self.do_epoch(epoch)
            self._print_debug(f'Epoch {epoch} ended in {timeit.default_timer() - start_epoch_time:.2f} sec\n',
                              verbose=1)

        self._print_debug(f'Master ended in {timeit.default_timer() - start_time:.2f} sec\n', verbose=0)

    def do_epoch(self, epoch):
        self._print_debug(f"Epoch {epoch}:", verbose=1)
        for it in range(1, self.epoch_len + 1):
            global_iter = (epoch - 1) * self.epoch_len + it

            # training each model one step (batch for ex.)
            for node_name, node in self.network.items():
                self.fit_step(node, epoch)
                # Save stat each stat_step step
                if global_iter % self.stat_step == 0:
                    accuracy = self.calc_accuracy(node, self.test_loader)
                    node.save_accuracy(accuracy, global_iter)
                    node.save_loss(global_iter)

            # Consensus starting from the {self.update_params_epoch_start}th epoch
            # with a period of {self.update_params_period}
            if epoch >= self.update_params_epoch_start \
                    and global_iter % self.update_params_period == 0:
                for node_name, node in self.network.items():
                    node.ask_params()

                for node_name, node in self.network.items():
                    self.update_params(node, epoch, global_iter)
