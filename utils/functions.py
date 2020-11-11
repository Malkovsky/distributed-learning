import torch
from torch.autograd import Variable
import wide_resnet_submodule.config as cf
import numpy as np


def fit_batch_cifar(node, epoch: int, *args, **kwargs):
    """
    Train node.model on one part of data which take from node.train_loader.
    :param node: node of ConsensusNode
    :param epoch: epoch number
    :param args: other unnamed params
    :param kwargs: other named params
    :return: nothing
    """
    images, labels = next(node.train_loader)

    node.model.train()
    node.model.training = True
    optimizer = node.optimizer(node.model.parameters(),
                               lr=cf.learning_rate(node.lr, epoch),
                               **node.opt_kwargs)
    train = Variable(images)
    labels = Variable(labels)

    # Clear gradients
    optimizer.zero_grad()

    # Forward propagation
    outputs = node.model(train)

    # Calculate softmax and ross entropy loss
    loss = node.error(outputs, labels)

    # Calculating gradients
    loss.backward()

    # Update parameters
    optimizer.step()

    node.loss_cum += loss.item()


def calc_accuracy_cifar(node, test_loader, *args, **kwargs):
    """
    Calculate node.model accuracy on data from test_loader
    :param node: node of ConsensusNode
    :param test_loader: something iterable
    :param args: other unnamed params
    :param kwargs: other named params
    :return: float accuracy
    """
    correct = 0
    total = 0

    node.model.eval()
    node.model.training = False

    with torch.no_grad():
        # Predict test dataset
        for images, labels in test_loader:
            test, labels = Variable(images), Variable(labels)

            # Forward propagation
            outputs = node.model(test)

            # Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]

            # Total number of labels
            total += len(labels)

            # Total correct predictions
            correct += predicted.eq(labels.data).cpu().sum()
    accuracy = 100 * correct / float(total)
    return float(accuracy)


def update_params_cifar(node, epoch: int, *args, **kwargs):
    # TODO: добавить зависимость коэф-та (сейчас 1.0) от номера эпохи
    """
    Update node.model.parameters using node.weights based on node.neighbors.
    :param node: node of ConsensusNode
    :param epoch: epoch number
    :param args: other unnamed params
    :param kwargs: other named params
    :return: nothing
    """
    for p in node.model.parameters():
        p.data *= node.weights[node.name]

    for node_name, params in node.parameters.items():
        for p, pn in zip(node.model.parameters(), params):
            p.data += pn.data * node.weights[node_name]


def fit_step_titanic(node, *args, **kwargs):
    """
    Train node.model on one part of data which take from node.train_loader.
    :param node: node of ConsensusNode
    :param args: other unnamed params
    :param kwargs: other named params
    :return: nothing
    """
    x_train, y_train = next(node.train_loader)
    train_loss = node.model.fit(x_train, y_train)
    node.loss_cum += train_loss


def calc_accuracy_titanic(node, test_loader, *args, **kwargs):
    """
    Calculate node.model accuracy on data from test_loader
    :param node: node of ConsensusNode
    :param test_loader: pair of x_test and y_test
    :param args: other unnamed params
    :param kwargs: other named params
    :return: float accuracy
        """
    x_test, y_test = test_loader
    return node.model.calc_accuracy(x_test, y_test)


def calc_loss_titanic(node, test_loader, *args, **kwargs):
    """
    Calculate node.model loss on data from test_loader
    :param node: node of ConsensusNode
    :param test_loader: pair of x_test and y_test
    :param args: other unnamed params
    :param kwargs: other named params
    :return: float loss
        """
    x_test, y_test = test_loader
    return node.model.calc_loss(x_test, y_test)


def update_params_titanic(node, epoch: int, *args, **kwargs):
    """
    Update node.model.parameters using node.weights based on node.neighbors.
    :param node: node of ConsensusNode
    :param epoch: epoch number
    :param args: other unnamed params
    :param kwargs: other named params
    :return: nothing
    """
    node.model.W *= node.weights[node.name]

    for node_name, params in node.parameters.items():
        node.model.W += params * node.weights[node_name]