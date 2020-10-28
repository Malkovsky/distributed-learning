from torch.autograd import Variable
from itertools import cycle
import torch
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
import utils.config as cf
from notebooks.networks import *


class ConsensusNode:
    def __init__(self, name: str, model, model_args: list, optimizer, optimizer_kwargs: dict, error,
                 train_loader, test_loader, weights: dict,
                 lr=0.02, stat_step=50, neighbors=None):
        self.model_name = model
        self.model = self._get_model(self.model_name)(*model_args)
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr

        self.error = error()
        self.train_loader = train_loader
        self.train_loader_iter = cycle(iter(self.train_loader))
        self.test_loader = deepcopy(test_loader)
        self.test_loader_iter = cycle(iter(self.test_loader))

        self.name: str = name
        self.neighbors: dict = neighbors
        self.weights: dict = weights
        self.parameters: dict = dict()

        self.curr_iter: int = 0
        self.train_loss: int = 0

        self.stat_step: int = stat_step
        self.accuracy_list: list = []
        self.iter_list: list = []
        self.loss_list: list = []

    def _get_model(self, model_name):
        if model_name == 'lenet':
            return LeNet
        elif model_name == 'ann':
            return ANNModel
        elif model_name == 'vggnet':
            return VGG
        elif model_name == 'resnet':
            return ResNet
        elif model_name == 'wide-resnet':
            return Wide_ResNet
        else:
            print("Error: Bad model name. Network should be either [ANN / LeNet / VGGNet / ResNet / Wide_ResNet",
                  file=sys.stderr)
            exit(0)

    def _calc_accuracy(self):
        correct = 0
        total = 0
        if self.model == 'ann':
            pass
        else:
            self.model.eval()
            self.model.training = False

        with torch.no_grad():
            # Predict test dataset
            for images, labels in self.test_loader:
                if self.model_name == 'ann':
                    test = Variable(images.view(-1, 28 * 28))
                else:
                    test, labels = Variable(images), Variable(labels)

                # Forward propagation
                outputs = self.model(test)

                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]

                # Total number of labels
                total += len(labels)

                # Total correct predictions
                if self.model_name == 'ann':
                    correct += (predicted == labels).sum()
                else:
                    correct += predicted.eq(labels.data).cpu().sum()

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

    def fit_step(self, epoch):
        self.curr_iter += 1

        images, labels = next(self.train_loader_iter)

        if self.model_name == 'ann':
            optimizer = self.optimizer(self.model.parameters(),
                                       lr=self.lr,
                                       **self.optimizer_kwargs)
            # Getting next batch
            train = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)
        else:
            self.model.train()
            self.model.training = True
            optimizer = self.optimizer(self.model.parameters(),
                                       lr=cf.learning_rate(self.lr, epoch),
                                       **self.optimizer_kwargs)
            train = Variable(images)
            labels = Variable(labels)

        # Clear gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = self.model(train)

        # Calculate softmax and ross entropy loss
        loss = self.error(outputs, labels)

        # Calculating gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        self.train_loss += loss.item()

        # Save stats
        if self.curr_iter % self.stat_step == 0:
            self.iter_list.append(self.curr_iter)
            self.loss_list.append(float(self.train_loss))
            self.train_loss = 0
            self.accuracy_list.append(float(self._calc_accuracy()))
            print(f"Epoch: {epoch}, Step: {self.curr_iter}, Node {self.name}:"
                  f" accuracy {self.accuracy_list[-1]:.2f}, loss {self.loss_list[-1]:.2f}")
