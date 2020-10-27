from torch.autograd import Variable
from itertools import cycle
import torch
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt


class ConsensusNode:
    def __init__(self, name: str, model, model_args: list, optimizer, error, train_loader, test_loader, weights: dict,
                 lr=0.02, niter=100, stat_step=50, neighbors=None):
        self.model = model(*model_args)
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.error = error()
        self.train_loader = train_loader
        self.train_loader_iter = cycle(iter(self.train_loader))
        self.test_loader = deepcopy(test_loader)
        self.test_loader_iter = cycle(iter(self.test_loader))

        self.name: str = name
        self.neighbors: dict = neighbors
        self.weights: dict = weights
        self.parameters: dict = dict()

        self.niter = niter
        self.curr_iter = 0

        self.stat_step = stat_step
        self.accuracy_list = []
        self.iter_list = []
        self.loss_list = []

    def _calc_accuracy(self):
        correct = 0
        total = 0

        # Predict test dataset
        for images, labels in self.test_loader:
            test = Variable(images.view(-1, 28 * 28))

            # Forward propagation
            outputs = self.model(test)

            # Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]

            # Total number of labels
            total += len(labels)

            # Total correct predictions
            correct += (predicted == labels).sum()

        return 100 * correct / float(total)

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors

    def get_params(self):
        return self.model.parameters()

    def show_graphs(self):
        fig, axs = plt.subplots(figsize=(20, 8), ncols=2)
        fig.suptitle(f'{self.name}', fontsize=24)
        fig.tight_layout(pad=4.0)
        sns.lineplot(x=self.iter_list, y=self.accuracy_list, ax=axs[0])
        axs[0].set_xlabel('Iteration', fontsize=16)
        axs[0].set_ylabel('Accuracy', fontsize=16)
        sns.lineplot(x=self.iter_list, y=self.loss_list, ax=axs[1])
        axs[1].set_xlabel('Iteration', fontsize=16)
        axs[1].set_ylabel('Loss', fontsize=16)

    def ask_params(self):
        self.parameters = {node_name: node.get_params()
                           for node_name, node in self.neighbors.items()}

    def update_params(self):
        for p in self.model.parameters():
            p.data *= self.weights[self.name]

        for node_name, params in self.parameters.items():
            for p, pn in zip(self.model.parameters(), params):
                p.data += pn.data * self.weights[node_name]

    def fit_step(self):
        self.curr_iter += 1

        # Getting next batch
        images, labels = next(self.train_loader_iter)
        train = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        # Clear gradients
        self.optimizer.zero_grad()

        # Forward propagation
        outputs = self.model(train)

        # Calculate softmax and ross entropy loss
        loss = self.error(outputs, labels)

        # Calculating gradients
        loss.backward()

        # Update parameters
        self.optimizer.step()

        # Save stats
        if self.curr_iter % self.stat_step == 0:
            self.iter_list.append(self.curr_iter)
            self.loss_list.append(float(loss.data))
            self.accuracy_list.append(float(self._calc_accuracy()))
            print(f"Step: {self.curr_iter}, Node {self.name}:"
                  f" accuracy {self.accuracy_list[-1]:.2f}, loss {self.loss_list[-1]:.2f}")
