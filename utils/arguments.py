import yaml


class Arguments():
    def __init__(self):
        # Data parameters
        self.data_path : str
        self.keyframes_path : str

        # Training parameters
        self.device : str
        self.epochs : int
        self.lr : float
        self.wd : float
        self.epsilon : float
        self.batch_size : int
        self.sequence_length : int
        self.weight_file : str
        self.log_file : str
        self.weight_decay : bool
        self.train_sequences : list
        self.precomputed_flow: bool
        self.stage : int
        self.w : int

    @classmethod
    def get_arguments(cls, config_path="utils/config.yaml"):
        args = yaml.load(open(config_path, "r"), yaml.Loader)
        return args