import torch
from torch.autograd import Variable
import wide_resnet_submodule.config as cf
import numpy as np
from itertools import chain


def fit_batch_cifar(master_node, node, epoch: int, *args, use_cuda=False, **kwargs):
    """
    Train node.model on one part of data which take from node.train_loader.
    :param master_node: node of MasterNode
    :param node: node of ConsensusNode
    :param epoch: epoch number
    :param use_cuda: set True to use CUDA
    :param args: other unnamed params
    :param kwargs: other named params
    :return: nothing
    """
    images, labels = next(node.train_loader)
    if use_cuda:
        images, labels = images.cuda(), labels.cuda()

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
    master_node.statistics['cumulative_train_loss'][node.name]['tmp'] += loss.item()


def get_cumulative_train_loss(master_node, node, *args, **kwargs):
    loss = master_node.statistics['cumulative_train_loss'][node.name]['tmp']
    master_node.statistics['cumulative_train_loss'][node.name]['tmp'] = 0.0
    return float(loss)


def calc_accuracy_cifar(master_node, node, *args, use_cuda=False, **kwargs):
    """
    Calculate node.model accuracy on data from test_loader
    :param master_node: node of MasterNode
    :param node: node of ConsensusNode
    :param use_cuda: set True to use CUDA
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
        for images, labels in master_node.test_loader:
            test, labels = Variable(images), Variable(labels)
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()

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


def get_flat_params_cifar(master_node, node, *args, use_cuda=False, **kwargs):
    """
    Get flattened array of model weights.
    :param master_node: node of MasterNode
    :param node: node of ConsensusNode
    :param args: other unnamed params
    :param kwargs: other named params
    :return: np.array parameters
    """
    if use_cuda:
        return np.array(list(chain.from_iterable([p.data.cpu().flatten() for p in node.get_params()])))
    return np.array(list(chain.from_iterable([p.data.flatten() for p in node.get_params()])))


def fit_step_titanic(master_node, node, *args, **kwargs):
    """
    Train node.model on one part of data which take from node.train_loader.
    :param master_node: node of MasterNode
    :param node: node of ConsensusNode
    :param args: other unnamed params
    :param kwargs: other named params
    :return: nothing
    """
    x_train, y_train = next(node.train_loader)
    train_loss = node.model.fit(x_train, y_train)
    master_node.statistics['cumulative_train_loss'][node.name]['tmp'] += train_loss


def calc_accuracy_titanic(master_node, node, *args, **kwargs):
    """
    Calculate node.model accuracy on data from test_loader
    :param master_node: node of MasterNode
    :param node: node of ConsensusNode
    :param args: other unnamed params
    :param kwargs: other named params
    :return: float accuracy
        """
    x_test, y_test = master_node.test_loader
    return node.model.calc_accuracy(x_test, y_test)


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


def get_flat_params_titanic(master_node, node, *args, **kwargs):
    """
    Get flattened array of model weights.
    :param master_node: node of MasterNode
    :param node: node of ConsensusNode
    :param args: other unnamed params
    :param kwargs: other named params
    :return: np.array parameters
    """
    return node.get_params()
