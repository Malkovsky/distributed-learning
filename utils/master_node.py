from utils.consensus_node import ConsensusNode


class MasterNode:
    def __init__(self, node_names, model, model_args: list, optimizer, optimizer_kwargs: dict,  error, weights: dict,
                 train_loaders: dict, test_loader,
                 stat_step=50,
                 lr=0.02,
                 epoch=200, epoch_len=391, epoch_cons_num=0):
        self.node_names = node_names
        self.model = model
        self.model_args = model_args
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr
        self.error = error
        self.weights: dict = weights
        self.network = None

        self.train_loaders = train_loaders
        self.test_loader = test_loader

        self.stat_step = stat_step
        self.epoch: int = epoch
        self.epoch_len: int = epoch_len
        self.epoch_cons_num: int = epoch_cons_num

    def initialize_nodes(self):
        self.network = {name: ConsensusNode(name=name,
                                            model=self.model,
                                            model_args=self.model_args,
                                            optimizer=self.optimizer,
                                            optimizer_kwargs=self.optimizer_kwargs,
                                            error=self.error,
                                            train_loader=self.train_loaders[name],
                                            test_loader=self.test_loader,
                                            lr=self.lr,
                                            stat_step=self.stat_step,
                                            weights=self.weights[name])
                        for name in self.node_names}

        for node_name, node in self.network.items():
            node.set_neighbors({neighbor_name: self.network[neighbor_name]
                                for neighbor_name in self.weights[node_name]
                                if neighbor_name != node_name})

    def start_consensus(self):
        for ep in range(1, self.epoch + 1):
            for it in range(self.epoch_len):
                for node_name, node in self.network.items():
                    node.fit_step(ep)
                if ep >= self.epoch_cons_num:
                    for node_name, node in self.network.items():
                        node.ask_params()

                    for node_name, node in self.network.items():
                        node.update_params()
