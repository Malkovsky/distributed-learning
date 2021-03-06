import pickle5 as pickle
import os


class Meter(object):

    def __init__(self, name, logger, load_path=None, save_path=''):

        self.name = name
        self.logger = logger
        self.value_history = []
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if load_path is not None:
            with open(load_path, 'rb') as f:
                init_dict = pickle.loads(f.read())
            for key in init_dict:
                try:
                    self.__dict__[key] = init_dict[key]
                except Exception:
                    logger.error('(Warning) Invalid key {} in loaded dict'.format(key))
            logger.info('Meter with dict {} loaded from {}'.format(init_dict, load_path))

    def add(self, val):
        self.value_history.append(val)
        self.logger.info('Meter {} add value {}'.format(self.name, val))
        return self

    def save(self):
        path = self.save_path + self.name + '.pickle'
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
        return self

    def get_value_history(self):
        return self.value_history
