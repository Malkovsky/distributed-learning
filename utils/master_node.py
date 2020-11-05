from utils.consensus_node import ConsensusNode
import sys


class MasterNode:
    # TODO: сплит данных на куски для каждой ноды
    # TODO: умный выбор весов from fast averaging
    def __init__(self,
                 node_names,
                 weights: dict,
                 train_loaders: dict,
                 test_loader,
                 stat_step=50,
                 lr=0.02,
                 epoch=200,
                 epoch_len=391,
                 epoch_cons_num=0,
                 verbose=0):
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
        self.weights: dict = weights
        self.network = None

        self.train_loaders = train_loaders
        self.test_loader = test_loader

        self.stat_step = stat_step
        self.epoch: int = epoch
        self.epoch_len: int = epoch_len
        self.epoch_cons_num: int = epoch_cons_num

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
                                            train_loader=self.train_loaders[name],
                                            test_loader=self.test_loader,
                                            lr=self.lr,
                                            stat_step=self.stat_step,
                                            weights=self.weights[name],
                                            verbose=self.verbose)
                        for name in self.node_names}

        for node_name, node in self.network.items():
            node._set_model(self.model, *self.model_args, **self.model_kwargs)
            node._set_optimizer(self.optimizer, *self.opt_args, **self.opt_kwargs)
            node._set_error(self.error, *self.error_args, **self.error_kwargs)
            node.set_neighbors({neighbor_name: self.network[neighbor_name]
                                for neighbor_name in self.weights[node_name]
                                if neighbor_name != node_name})
        return self

    def start_consensus(self):
        # TODO: изначально брать одинаковые веса?
        # TODO: замерять время одной эпохи
        for ep in range(1, self.epoch + 1):
            for it in range(self.epoch_len):
                for node_name, node in self.network.items():
                    node.fit_step(ep)
                if ep >= self.epoch_cons_num:
                    for node_name, node in self.network.items():
                        node.ask_params()

                    for node_name, node in self.network.items():
                        node.update_params()
