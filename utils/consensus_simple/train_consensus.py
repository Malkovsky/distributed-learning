import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import numpy as np
import time

from networks.resnet import resnet20
from utils.consensus_simple.meter import Meter
from utils.consensus_simple.mixer import Mixer
from utils.consensus_simple.utils import *

SEED = 42

DATASET_MEAN = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

DATASET_STD = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}


def get_cifar10_train_loaders(args):
    n_agents = args['n_agents']

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(DATASET_MEAN['cifar10'], DATASET_STD['cifar10']),
    ])
    trainset = torchvision.datasets.CIFAR10(root=args['dataset_dir'],
                                            train=True, download=True,
                                            transform=transform_train)

    indices = [i for i in range(len(trainset))]
    np.random.shuffle(indices)
    indices = indices[:n_agents * (len(trainset) // n_agents)]
    indices = np.array_split(indices, n_agents)
    subsets = [torch.utils.data.Subset(trainset, indices=ind) for ind in indices]

    train_loaders = []
    for i in range(n_agents):
        train_loaders.append(torch.utils.data.DataLoader(subsets[i],
                                                         batch_size=args['train_batch_size'],
                                                         shuffle=True,
                                                         num_workers=2))
    return train_loaders


def get_cifar10_test_loader(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(DATASET_MEAN[args['dataset_name']], DATASET_STD[args['dataset_name']]),
    ])

    testset = torchvision.datasets.CIFAR10(root=args['dataset_dir'],
                                           train=False, download=False,
                                           transform=transform_test)

    return torch.utils.data.DataLoader(testset, batch_size=args['test_batch_size'], shuffle=False, num_workers=2)


def get_resnet20_models(args):
    topology = args['topology']
    models = {}

    main_state_dict = None

    for agent in topology:
        models[agent] = args['model']()
        if args['equalize_start_params']:
            if main_state_dict is None:
                main_state_dict = models[agent].state_dict()
            else:
                models[agent].load_state_dict(main_state_dict)

        models[agent] = models[agent].cuda()
        models[agent] = torch.nn.DataParallel(models[agent], device_ids=range(torch.cuda.device_count()))

    return models


def get_criterions(args):
    return {agent: nn.CrossEntropyLoss().cuda() for agent in args['topology']}


def get_optimizers(args, models):
    return {agent: torch.optim.SGD(models[agent].parameters(),
                                   lr=args['lr'],
                                   momentum=args['momentum'],
                                   weight_decay=args['weight_decay'])
            for agent in args['topology']}


def get_lr_schedulers(args, optimizers, lr_schedule):
    return {agent: torch.optim.lr_scheduler.LambdaLR(optimizers[agent],
                                                     lr_lambda=lr_schedule)
            for agent in args['topology']}


def fit_batch(model, optimizer, criterion, train_iter):
    # fit model on one batch
    inputs, targets = train_iter.next()
    inputs, targets = inputs.cuda(), targets.cuda()

    optimizer.zero_grad()
    outputs = model(inputs)  # Forward Propagation
    loss = criterion(outputs, targets)  # Loss
    loss.backward()  # Backward Propagation
    optimizer.step()  # Optimizer update

    _, predicted = torch.max(outputs.data, 1)

    return {'loss': loss.item(),
            'total': targets.size(0),
            'correct': predicted.eq(targets.data).cpu().sum()}


def fit_epoch(args, models, train_loaders, criterions, optimizers, epoch, mixer, logger):
    # train one epoch
    topology = args['topology']

    for agent in topology:
        models[agent].train()
        models[agent].training = True

    train_iters = {agent: iter(train_loaders[agent]) for agent in topology}

    logger.info('\nStart Training Epoch #{}'.format(epoch))
    for agent in topology:
        logger.info('Agent {}, optimizer LR={:.4f}'.format(agent, optimizers[agent].state_dict()['param_groups'][0]['lr']))

    iterations = len(train_loaders[0])

    epoch_stats = {agent: {'loss': 0.0,
                           'total': 0,
                           'correct': 0}
                   for agent in topology}

    for it in range(iterations):
        for agent in topology:
            stats = fit_batch(models[agent], optimizers[agent], criterions[agent], train_iters[agent])

            for key, value in stats.items():
                epoch_stats[agent][key] += value

        # consensus averaging with {args['consensus_freq']} frequency
        global_iteration = iterations*(epoch - 1) + it
        if global_iteration % args['consensus_freq'] == 0:
            mixer.mix(times=args['consensus_times'], eps=args['consensus_eps'])

    return {agent: {'loss': epoch_stats[agent]['loss'],
                    'accuracy': 100.*epoch_stats[agent]['correct']/epoch_stats[agent]['total']}
            for agent in topology}


def test(model, test_loader, criterion):
    model.eval()
    model.training = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        acc = 100.*correct/total

    return {'accuracy': acc,
            'loss': test_loss}


def main(args):

    elapsed_time = 0
    start_time = time.time()

    # parse arguments, filling in missing values
    basic_args = {
        'lr': 0.1,
        'dataset_name': 'cifar10',
        'dataset_dir': '../data/cifar10',
        'train_batch_size': 32,
        'test_batch_size': 128,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'topology': {0: {0: 0.6, 1: 0.4}, 1: {0: 0.4, 1: 0.6}},
        'n_agents': 2,
        'equalize_start_params': True,
        'use_lsr': True,
        'num_epochs': 2,
        'consensus_freq': 1,
        'consensus_eps': None,
        'consensus_times': 1,
        'log_file_path': 'logs.log',
        'meters_path': 'meters/',
        'model': resnet20,
    }
    for key, value in basic_args.items():
        if key not in args:
            args[key] = value

    topology = args['topology']

    def lr_schedule(epoch):
        factor = args['n_agents'] if args['use_lsr'] else 1.0
        if epoch >= 81:
            factor /= 10
        if epoch >= 122:
            factor /= 10
        return factor

    if 'lr_schedule' not in args:
        args['lr_schedule'] = lr_schedule

    if 'logger' in args:
        logger = args['logger']
    else:
        logger = get_logger(log_file_path=args['log_file_path'])
    logger.info('START with args \n{}'.format(args))

    # preparing
    train_loaders = get_cifar10_train_loaders(args)
    logger.info('{} Train loaders successfully prepared'.format(len(train_loaders)))

    test_loader = get_cifar10_test_loader(args)
    logger.info('Test loader with length {} successfully prepared'.format(len(test_loader)))

    criterions = get_criterions(args)
    logger.info('{} Criterions successfully prepared'.format(len(criterions)))

    models = get_resnet20_models(args)
    logger.info('{} Models successfully prepared'.format(len(models)))

    optimizers = get_optimizers(args, models)
    logger.info('{} Optimizers successfully prepared'.format(len(optimizers)))

    lr_schedulers = get_lr_schedulers(args, optimizers, lr_schedule=lr_schedule)
    logger.info('{} LR schedulers successfully prepared'.format(len(lr_schedulers)))

    meters = {
        agent: {
            'train_loss': Meter('{}_train_loss'.format(agent), logger, save_path=args['meters_path']),
            'train_accuracy': Meter('{}_train_accuracy'.format(agent), logger, save_path=args['meters_path']),
            'test_loss': Meter('{}_test_loss'.format(agent), logger, save_path=args['meters_path']),
            'test_accuracy': Meter('{}_test_accuracy'.format(agent), logger, save_path=args['meters_path']),
            'params_deviation': Meter('{}_params_deviation'.format(agent), logger, save_path=args['meters_path']),
        }
        for agent in topology
    }
    meters['epoch_time'] = Meter('epoch_time', logger, save_path=args['meters_path'])
    logger.info('Meters successfully prepared')

    mixer = Mixer(models, topology, logger)
    logger.info('Mixer successfully prepared')

    elapsed_time += time.time() - start_time
    logger.info('Preparing took {}:{:02d}:{:02d}'.format(*get_hms(elapsed_time)))

    # start training
    for epoch in range(1, args['num_epochs'] + 1):
        start_time = time.time()

        train_stats = fit_epoch(args, models, train_loaders, criterions, optimizers, epoch, mixer, logger)
        param_dev_stats = mixer.get_parameters_deviation()

        for agent in topology:
            # saving train statistics
            meters[agent]['train_loss'].add(train_stats[agent]['loss']).save()
            meters[agent]['train_accuracy'].add(train_stats[agent]['accuracy']).save()

            # saving parameters deviation statistics
            meters[agent]['params_deviation'].add(param_dev_stats[agent]).save()

            # update lr by lr_schedule
            lr_schedulers[agent].step()

            # saving test statistics
            test_stats = test(models[agent], test_loader, criterions[agent])
            meters[agent]['test_loss'].add(test_stats['loss']).save()
            meters[agent]['test_accuracy'].add(test_stats['accuracy']).save()

        epoch_time = time.time() - start_time
        meters['epoch_time'].add(epoch_time).save()
        elapsed_time += epoch_time
        logger.info('Elapsed time : {}:{:02d}:{:02d}'.format(*get_hms(elapsed_time)))

    logger.info('FINISH')


if __name__ == '__main__':
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    #tmp_test_meter()
    #tmp_test_mixer()
    main({})
