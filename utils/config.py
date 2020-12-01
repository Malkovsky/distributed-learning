############### CIFAR-10 consensus configuration file ###############
import math


def weights_schedule_dummy_increase(weights, self_name, epoch, num_epoch, *args, **kwargs):
    if len(weights) == 1:
        return weights

    change_part = 1. - weights[self_name]
    coef = 0

    if epoch > 80:
        coef = 0.875
    elif epoch > 60:
        coef = 0.75
    elif epoch > 40:
        coef = 0.5

    new_weights = {name: (w + change_part * coef
                          if name == self_name
                          else w * (1. - coef)
                          )
                   for name, w in weights.items()
                   }
    return new_weights


def weights_schedule_dummy_decrease(weights, self_name, epoch, num_epoch, *args, **kwargs):
    if len(weights) == 1:
        return weights

    change_part = 1. - weights[self_name]

    if epoch > 80:
        coef = 0.
    elif epoch > 60:
        coef = 0.5
    elif epoch > 40:
        coef = 0.75
    else:
        coef = 0.9

    new_weights = {name: (w + change_part * coef
                          if name == self_name
                          else w * (1. - coef)
                          )
                   for name, w in weights.items()
                   }
    return new_weights


def weights_schedule_log_decrease(weights, self_name, epoch, num_epoch, *args, **kwargs):
    if len(weights) == 1:
        return weights

    change_part = 1. - weights[self_name]
    x = 1. + weights[self_name] + math.log(epoch, 1./(num_epoch / 2.))
    x = min(x, 1.)
    x = max(x, weights[self_name])

    coef = abs(x - weights[self_name]) / change_part

    new_weights = {name: (w + change_part * coef
                          if name == self_name
                          else w * (1. - coef)
                          )
                   for name, w in weights.items()
                   }
    return new_weights


def weights_schedule_log_increase(weights, self_name, epoch, num_epoch, *args, **kwargs):
    if len(weights) == 1:
        return weights

    change_part = 1. - weights[self_name]

    x = weights[self_name] - math.log(epoch, 1./(num_epoch*50.))
    x = min(x, 1.)
    x = max(x, weights[self_name])

    coef = abs(x - weights[self_name]) / change_part

    new_weights = {name: (w + change_part * coef
                          if name == self_name
                          else w * (1. - coef)
                          )
                   for name, w in weights.items()
                   }
    return new_weights


def weights_schedule_linear_decrease(weights, self_name, epoch, num_epoch, *args, **kwargs):
    if len(weights) == 1:
        return weights

    change_part = 1. - weights[self_name]

    k = 1 / (1.2*num_epoch)
    x = max(weights[self_name], 1 - epoch*k)

    coef = abs(x - weights[self_name]) / change_part

    new_weights = {name: (w + change_part * coef
                          if name == self_name
                          else w * (1. - coef)
                          )
                   for name, w in weights.items()
                   }
    return new_weights


def weights_schedule_linear_increase(weights, self_name, epoch, num_epoch, *args, **kwargs):
    if len(weights) == 1:
        return weights

    change_part = 1. - weights[self_name]

    k = 1 / (1.2*num_epoch)
    x = min(1., epoch*k)
    x = max(weights[self_name], x)

    coef = abs(x - weights[self_name]) / change_part

    new_weights = {name: (w + change_part * coef
                          if name == self_name
                          else w * (1. - coef)
                          )
                   for name, w in weights.items()
                   }
    return new_weights


def lr_schedule_default(lr, epoch, *args, **kwargs):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1

    return lr * math.pow(0.2, optim_factor)


def lr_schedule_smooth(lr, epoch, smooth_factor=50, *args, **kwargs):
    p = -1 + epoch / smooth_factor
    res = lr * pow(0.2, p)
    res = min(res, lr)
    return res


def lr_const(*args, const=0.02, **kwargs):
    return const


def lr_schedule_div3(lr, epoch, *args, **kwargs):
    optim_factor = 0
    if epoch > 60:
        optim_factor = 3
    elif epoch > 40:
        optim_factor = 2
    elif epoch > 20:
        optim_factor = 1

    return lr * math.pow(0.2, optim_factor)


if __name__ == '__main__':
    weights = {'Alice': {'Alice': 0.5, 'Bob': 0.25, 'Charlie': 0.25},
               'Bob': {'Alice': 0.25, 'Bob': 0.5, 'Charlie': 0.25},
               'Charlie': {'Alice': 0.25, 'Bob': 0.25, 'Charlie': 0.5}}

    """weights = {'Alice': {'Alice': 0.34, 'Bob': 0.33, 'Charlie': 0.33},
               'Bob': {'Alice': 0.33, 'Bob': 0.34, 'Charlie': 0.33},
               'Charlie': {'Alice': 0.33, 'Bob': 0.33, 'Charlie': 0.34}}"""
    schedule = weights_schedule_linear_decrease

    new_w = schedule(weights['Alice'], 'Alice', epoch=1, num_epoch=200)
    print(new_w, sum(new_w.values()))

    new_w = schedule(weights['Alice'], 'Alice', epoch=41, num_epoch=200)
    print(new_w, sum(new_w.values()))

    new_w = schedule(weights['Alice'], 'Alice', epoch=61, num_epoch=200)
    print(new_w, sum(new_w.values()))

    new_w = schedule(weights['Alice'], 'Alice', epoch=81, num_epoch=200)
    print(new_w, sum(new_w.values()))

    new_w = schedule(weights['Alice'], 'Alice', epoch=1, num_epoch=100)
    print(new_w, sum(new_w.values()))

    new_w = schedule(weights['Alice'], 'Alice', epoch=41, num_epoch=100)
    print(new_w, sum(new_w.values()))

    new_w = schedule(weights['Alice'], 'Alice', epoch=61, num_epoch=100)
    print(new_w, sum(new_w.values()))

    new_w = schedule(weights['Alice'], 'Alice', epoch=81, num_epoch=100)
    print(new_w, sum(new_w.values()))

    schedule = weights_schedule_linear_increase
    print('Increase type:')

    new_w = schedule(weights['Alice'], 'Alice', epoch=1, num_epoch=200)
    print(new_w, sum(new_w.values()))

    new_w = schedule(weights['Alice'], 'Alice', epoch=41, num_epoch=200)
    print(new_w, sum(new_w.values()))

    new_w = schedule(weights['Alice'], 'Alice', epoch=61, num_epoch=200)
    print(new_w, sum(new_w.values()))

    new_w = schedule(weights['Alice'], 'Alice', epoch=81, num_epoch=200)
    print(new_w, sum(new_w.values()))

    new_w = schedule(weights['Alice'], 'Alice', epoch=151, num_epoch=200)
    print(new_w, sum(new_w.values()))

    new_w = schedule(weights['Alice'], 'Alice', epoch=181, num_epoch=200)
    print(new_w, sum(new_w.values()))

    new_w = schedule(weights['Alice'], 'Alice', epoch=1, num_epoch=100)
    print(new_w, sum(new_w.values()))

    new_w = schedule(weights['Alice'], 'Alice', epoch=41, num_epoch=100)
    print(new_w, sum(new_w.values()))

    new_w = schedule(weights['Alice'], 'Alice', epoch=61, num_epoch=100)
    print(new_w, sum(new_w.values()))

    new_w = schedule(weights['Alice'], 'Alice', epoch=81, num_epoch=100)
    print(new_w, sum(new_w.values()))