############### CIFAR-10 consensus configuration file ###############

def weights_schedule(weights, self_name, epoch, schedule=None):
    if not schedule:
        return weights

    change_part = 1. - weights[self_name]
    coef = 0

    if schedule == 'increase':
        if epoch > 80:
            coef = 0.875
        elif epoch > 60:
            coef = 0.75
        elif epoch > 40:
            coef = 0.5
    elif schedule == 'decrease':
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


if __name__ == '__main__':
    weights = {'Alice': {'Alice': 0.5, 'Bob': 0.25, 'Charlie': 0.25},
               'Bob': {'Alice': 0.25, 'Bob': 0.5, 'Charlie': 0.25},
               'Charlie': {'Alice': 0.25, 'Bob': 0.25, 'Charlie': 0.5}}

    weights = {'Alice': {'Alice': 0.34, 'Bob': 0.33, 'Charlie': 0.33},
               'Bob': {'Alice': 0.33, 'Bob': 0.34, 'Charlie': 0.33},
               'Charlie': {'Alice': 0.33, 'Bob': 0.33, 'Charlie': 0.34}}
    print('Increase type:')

    new_w = weights_schedule(weights['Alice'], 'Alice', epoch=1)
    print(new_w, sum(new_w.values()))

    new_w = weights_schedule(weights['Alice'], 'Alice', epoch=41)
    print(new_w, sum(new_w.values()))

    new_w = weights_schedule(weights['Alice'], 'Alice', epoch=61)
    print(new_w, sum(new_w.values()))

    new_w = weights_schedule(weights['Alice'], 'Alice', epoch=81)
    print(new_w, sum(new_w.values()))

    print('Decrease type:')

    new_w = weights_schedule(weights['Alice'], 'Alice', epoch=1, schedule='decrease')
    print(new_w, sum(new_w.values()))

    new_w = weights_schedule(weights['Alice'], 'Alice', epoch=41, schedule='decrease')
    print(new_w, sum(new_w.values()))

    new_w = weights_schedule(weights['Alice'], 'Alice', epoch=61, schedule='decrease')
    print(new_w, sum(new_w.values()))

    new_w = weights_schedule(weights['Alice'], 'Alice', epoch=81, schedule='decrease')
    print(new_w, sum(new_w.values()))
