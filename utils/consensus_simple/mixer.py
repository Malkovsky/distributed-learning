import torch
import numpy as np


def basic_deviation_metric(p1, p2):
    return np.linalg.norm(p1 - p2)


class Mixer(object):
    def __init__(self, models, topology, logger, dev_metric=None):
        self.models = models
        self.topology = topology
        self.logger = logger
        self.dev_metric = basic_deviation_metric
        if dev_metric is not None:
            self.dev_metric = dev_metric

    def mix(self, times=1, eps=None):
        if len(self.topology) <= 1:
            return 0

        self.logger.debug('Mixer start with times= {}, eps= {}'.format(times, eps))

        with torch.no_grad():
            times_done = 0
            agents_params = {agent: self._get_flatten_model_params(self.models[agent]) for agent in self.topology}

            stopping_criterion = self._update_stopping_criterion(agents_params, times_done, times, eps)
            while not stopping_criterion:
                agents_params = self._mix_params_once(agents_params)
                times_done += 1
                stopping_criterion = self._update_stopping_criterion(agents_params, times_done, times, eps)

            for agent in self.topology:
                self._load_flatten_params_to_model(self.models[agent], agents_params[agent])

        self.logger.debug('Mixer finished with {} times'.format(times_done))
        return times_done

    def _update_stopping_criterion(self, agent_parameters, times_done, max_times, eps):
        return (eps is None or self._get_max_deviation(agent_parameters) < eps) and (times_done >= max_times)

    def _mix_params_once(self, params):
        mixed_params = {}

        for agent in self.topology:
            mixed_params[agent] = sum(params[neighbor] * weight for neighbor, weight in self.topology[agent].items())

        return mixed_params

    def _get_max_deviation(self, params):
        devs_list = self._get_deviation_dict(params).values()
        max_dev = max(devs_list)
        self.logger.debug('Mixer calculate max deviation= {}'.format(max_dev))
        return max_dev

    def _get_deviation_dict(self, params):
        if len(self.topology) <= 1:
            return {agent: 0.0 for agent in self.topology}
        devs = {}
        for agent in self.topology:
            avg_neighbors_params = np.mean([params[neighbor] for neighbor in self.topology[agent] if neighbor != agent],
                                           axis=0)
            devs[agent] = self.dev_metric(params[agent], avg_neighbors_params)
        return devs

    def _get_flatten_model_params(self, model):
        return torch.cat([p.data.to(torch.float32).view(-1) for p in model.parameters()]).detach().clone().cpu().numpy()

    def  _load_flatten_params_to_model(self, model, params):
        used_params = 0
        for p in model.parameters():
            cnt_params = p.numel()
            p.data.copy_(torch.Tensor(params[used_params:used_params + cnt_params]).view(p.shape).to(p.dtype))
            used_params += cnt_params

    def get_parameters_deviation(self):
        agents_params = {agent: self._get_flatten_model_params(self.models[agent]) for agent in self.topology}
        return self._get_deviation_dict(agents_params)
