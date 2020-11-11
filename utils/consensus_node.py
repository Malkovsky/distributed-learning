import seaborn as sns
import matplotlib.pyplot as plt
from wide_resnet_submodule.networks import *


class ConsensusNode:
    # TODO: поддержка  GPU
    # TODO: сохранение модели в память
    def __init__(self,
                 name: str,
                 weights: dict,
                 train_loader,
                 lr=0.02,
                 stat_step=50,
                 verbose=0):
        """
        Class implementing consensus node in consensus network.
        :param name: unique node name in consensus network
        :param weights: dict of node names and weights corresponding edges
        :param train_loader: generator of train batches
        :param lr: gradient learning rate
        :param stat_step: period of statistic save
        :param verbose: verbose mode
        """
        self.model = None
        self.optimizer = None
        self.opt_args = None
        self.opt_kwargs = None
        self.error = None

        self.lr = lr

        self.train_loader = train_loader

        self.name: str = name
        self.weights: dict = weights
        self.parameters: dict = dict()
        self.neighbors: dict = dict()

        self.curr_iter: int = 0
        self.loss_cum: int = 0

        self.stat_step: int = stat_step
        self.accuracy_list: list = [[], []]
        self.loss_list: list = [[], []]

        self.verbose = verbose
        self.debug_file = sys.stdout

    def save_accuracy(self, accuracy, iteration):
        """
        Saves model accuracy
        :param accuracy: float accuracy
        :param iteration: iteration number
        :return: self
        """
        self.accuracy_list[0].append(accuracy)
        self.accuracy_list[1].append(iteration)
        self._print_debug(f"Node {self.name}: iter {iteration}, accuracy= {accuracy:.2f}", verbose=1)
        return self

    def save_loss(self, it):
        """
        Saves cumulative model loss and zeroing this
        :param it:
        :return:
        """
        self.loss_list[0].append(self.loss_cum)
        self.loss_list[1].append(it)
        self._print_debug(f"Node {self.name}: iter {it}, cumulative train loss= {self.loss_cum:.2f}", verbose=1)
        self.loss_cum = 0.
        return self

    def _print_debug(self, msg, verbose):
        """
        Print msg if good verbose mode
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
        self.model = model(*args, *kwargs)
        self._print_debug(f"Node {self.name} set model={self.model} with args={args}, kwargs={kwargs}", 2)
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
        self._print_debug(f"Node {self.name} set optimizer={self.optimizer} with args={args}, kwargs={kwargs}", 2)
        return self

    def set_error(self, error, *args, **kwargs):
        """
        Sets self error function on given func
        :param error: some error function
        :param args: other unnamed params
        :param kwargs: other named params
        :return: self
        """
        self.error = error(*args, **kwargs)
        self._print_debug(f"Node {self.name} set error={self.error} with args={args}, kwargs={kwargs}", 2)
        return self

    def set_neighbors(self, neighbors):
        """
        Sets node's neighbors on given neighbors
        :param neighbors: dict of nodes names and links
        :return:
        """
        self.neighbors = neighbors
        return self

    def get_params(self):
        """
        Returns model parameters
        :return: model parameters
        """
        return self.model.parameters()

    def show_graphs(self):
        """
        Shows all statistics
        :return: nothing
        """
        fig, axs = plt.subplots(figsize=(20, 8), ncols=2)
        fig.suptitle(f'{self.name}', fontsize=24)
        fig.tight_layout(pad=4.0)
        sns.lineplot(x=self.accuracy_list[1], y=self.accuracy_list[0], ax=axs[0])
        axs[0].set_xlabel('Iteration', fontsize=16)
        axs[0].set_ylabel('Accuracy', fontsize=16)
        sns.lineplot(x=self.loss_list[1], y=self.loss_list[0], ax=axs[1])
        axs[1].set_xlabel('Iteration', fontsize=16)
        axs[1].set_ylabel('Loss', fontsize=16)

    def ask_params(self):
        """
        Asks and saves neighbors model parameters
        :return: self
        """
        self.parameters = {node_name: node.get_params()
                           for node_name, node in self.neighbors.items()}
        return self
