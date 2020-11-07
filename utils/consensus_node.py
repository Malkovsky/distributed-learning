import seaborn as sns
import matplotlib.pyplot as plt
from wide_resnet_submodule.networks import *

model_type = {
    'lenet': LeNet,
    'vggnet': VGG,
    'resnet': ResNet,
    'wide-resnet': Wide_ResNet
}


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

    def _save_accuracy(self, accuracy, it):
        self.accuracy_list[0].append(accuracy)
        self.accuracy_list[1].append(it)
        self._print_debug(f"Node {self.name}: iter {it}, accuracy= {accuracy:.2f}", verbose=1)

    def _save_loss(self, it):
        self.loss_list[0].append(self.loss_cum)
        self.loss_list[1].append(it)
        self._print_debug(f"Node {self.name}: iter {it}, cumulative train loss= {self.loss_cum:.2f}", verbose=1)
        self.loss_cum = 0.

    def _print_debug(self, msg, verbose):
        if verbose <= self.verbose:
            print(msg, file=self.debug_file)

    def _set_model(self, model, *args, **kwargs):
        self.model = model(*args, *kwargs)
        self._print_debug(f"Node {self.name} set model={self.model} with args={args}, kwargs={kwargs}", 2)

    def _set_optimizer(self, optimizer, *args, **kwargs):
        self.optimizer = optimizer
        self.opt_args = args
        self.opt_kwargs = kwargs
        self._print_debug(f"Node {self.name} set optimizer={self.optimizer} with args={args}, kwargs={kwargs}", 2)

    def _set_error(self, error, *args, **kwargs):
        self.error = error(*args, **kwargs)
        self._print_debug(f"Node {self.name} set error={self.error} with args={args}, kwargs={kwargs}", 2)

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors

    def get_params(self):
        return self.model.parameters()

    def show_graphs(self):
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
        self.parameters = {node_name: node.get_params()
                           for node_name, node in self.neighbors.items()}

