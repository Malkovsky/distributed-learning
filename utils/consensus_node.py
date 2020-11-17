from wide_resnet_submodule.networks import *
import torch
import torch.backends.cudnn as cudnn


class ConsensusNode:
    def __init__(self,
                 name: str,
                 weights: dict,
                 train_loader,
                 lr=0.02,
                 use_cuda=False,
                 verbose=1):
        """
        Class implementing consensus node in consensus network.
        :param name: unique node name in consensus network
        :param weights: dict of node names and weights corresponding edges
        :param train_loader: generator of train batches
        :param lr: gradient learning rate
        :param use_cuda: set True to use CUDA
        :param verbose: verbose mode
        """
        self.model = None
        self.optimizer = None
        self.opt_args = None
        self.opt_kwargs = None
        self.error = None

        self.lr = lr

        self.train_loader = train_loader

        self.name: str = name
        self.weights: dict = weights
        self.parameters: dict = dict()
        self.neighbors: dict = dict()

        self.use_cuda = use_cuda

        self.verbose = verbose
        self.debug_file = sys.stdout

    def _print_debug(self, msg, verbose):
        """
        Print msg if good verbose mode
        :param msg: string of message
        :param verbose: verbose mode
        :return: self
        """
        if verbose <= self.verbose:
            print(msg, file=self.debug_file)
        return self

    def set_model(self, model,  *args, resume_path=None, **kwargs):
        """
        Sets self.model on given model
        :param model: some model with interface like models in pytorch
        :param resume_path: path to saved model
        :param args: other unnamed params
        :param kwargs: other named params
        :return: self
        """
        # load model from resume path if exists
        if resume_path:
            checkpoint = torch.load(resume_path)
            self.model = checkpoint['model']
            self._print_debug(f'Node {self.name} successfully loads the model {self.model} from {resume_path}.',
                              verbose=3)
            return self

        self.model = model(*args, *kwargs)
        if self.use_cuda:
            self.model.cuda()
            self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        self._print_debug(f"Node {self.name} set model= {self.model} with args= {args},"
                          f" kwargs= {kwargs}, use CUDA= {self.use_cuda}", 3)
        return self

    def set_optimizer(self, optimizer, *args, **kwargs):
        """
        Sets self optimizer on given optimizer
        :param optimizer: some pytorch optimizer
        :param args: other unnamed params
        :param kwargs: other named params
        :return: self
        """
        self.optimizer = optimizer
        self.opt_args = args
        self.opt_kwargs = kwargs
        self._print_debug(f"Node {self.name} set optimizer={self.optimizer} with args={args}, kwargs={kwargs}", 3)
        return self

    def set_error(self, error, *args, **kwargs):
        """
        Sets self error function on given func
        :param error: some error function
        :param args: other unnamed params
        :param kwargs: other named params
        :return: self
        """
        self.error = error(*args, **kwargs)
        self._print_debug(f"Node {self.name} set error={self.error} with args={args}, kwargs={kwargs}", 3)
        return self

    def set_neighbors(self, neighbors):
        """
        Sets node's neighbors on given neighbors
        :param neighbors: dict of nodes names and links
        :return:
        """
        self.neighbors = neighbors
        return self

    def get_params(self):
        """
        Returns model parameters
        :return: model parameters
        """
        return self.model.parameters()

    def ask_params(self):
        """
        Asks and saves neighbors model parameters
        :return: self
        """
        self.parameters = {node_name: node.get_params()
                           for node_name, node in self.neighbors.items()}
        return self
