############### CIFAR-10 consensus configuration file ###############
import math


def weights_schedule_dummy_increase(weights, self_name, epoch, num_epoch):
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


def weights_schedule_dummy_decrease(weights, self_name, epoch, num_epoch):
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


def weights_schedule_log_decrease(weights, self_name, epoch, num_epoch):
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


def weights_schedule_log_increase(weights, self_name, epoch, num_epoch):
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


if __name__ == '__main__':
    weights = {'Alice': {'Alice': 0.5, 'Bob': 0.25, 'Charlie': 0.25},
               'Bob': {'Alice': 0.25, 'Bob': 0.5, 'Charlie': 0.25},
               'Charlie': {'Alice': 0.25, 'Bob': 0.25, 'Charlie': 0.5}}

    """weights = {'Alice': {'Alice': 0.34, 'Bob': 0.33, 'Charlie': 0.33},
               'Bob': {'Alice': 0.33, 'Bob': 0.34, 'Charlie': 0.33},
               'Charlie': {'Alice': 0.33, 'Bob': 0.33, 'Charlie': 0.34}}"""
    print('Decrease type:')
    new_w = weights_schedule_log_decrease(weights['Alice'], 'Alice', epoch=1, num_epoch=200)
    print(new_w, sum(new_w.values()))

    new_w = weights_schedule_log_decrease(weights['Alice'], 'Alice', epoch=41, num_epoch=200)
    print(new_w, sum(new_w.values()))

    new_w = weights_schedule_log_decrease(weights['Alice'], 'Alice', epoch=61, num_epoch=200)
    print(new_w, sum(new_w.values()))

    new_w = weights_schedule_log_decrease(weights['Alice'], 'Alice', epoch=81, num_epoch=200)
    print(new_w, sum(new_w.values()))

    new_w = weights_schedule_log_decrease(weights['Alice'], 'Alice', epoch=1, num_epoch=100)
    print(new_w, sum(new_w.values()))

    new_w = weights_schedule_log_decrease(weights['Alice'], 'Alice', epoch=41, num_epoch=100)
    print(new_w, sum(new_w.values()))

    new_w = weights_schedule_log_decrease(weights['Alice'], 'Alice', epoch=61, num_epoch=100)
    print(new_w, sum(new_w.values()))

    new_w = weights_schedule_log_decrease(weights['Alice'], 'Alice', epoch=81, num_epoch=100)
    print(new_w, sum(new_w.values()))

    print('Increase type:')

    new_w = weights_schedule_log_increase(weights['Alice'], 'Alice', epoch=1, num_epoch=200)
    print(new_w, sum(new_w.values()))

    new_w = weights_schedule_log_increase(weights['Alice'], 'Alice', epoch=41, num_epoch=200)
    print(new_w, sum(new_w.values()))

    new_w = weights_schedule_log_increase(weights['Alice'], 'Alice', epoch=61, num_epoch=200)
    print(new_w, sum(new_w.values()))

    new_w = weights_schedule_log_increase(weights['Alice'], 'Alice', epoch=81, num_epoch=200)
    print(new_w, sum(new_w.values()))

    new_w = weights_schedule_log_increase(weights['Alice'], 'Alice', epoch=1, num_epoch=100)
    print(new_w, sum(new_w.values()))

    new_w = weights_schedule_log_increase(weights['Alice'], 'Alice', epoch=41, num_epoch=100)
    print(new_w, sum(new_w.values()))

    new_w = weights_schedule_log_increase(weights['Alice'], 'Alice', epoch=61, num_epoch=100)
    print(new_w, sum(new_w.values()))

    new_w = weights_schedule_log_increase(weights['Alice'], 'Alice', epoch=81, num_epoch=100)
    print(new_w, sum(new_w.values()))