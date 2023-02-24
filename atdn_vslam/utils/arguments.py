import yaml
import os

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
        self.augment_flow : bool
        self.train_sequences : list
        self.stage : int
        self.w : int

    @classmethod
    def get_arguments(cls, config_path=None):
        if config_path is None:
            current_path = os.path.realpath(__file__)
            head, _ = os.path.split(current_path)
            config_path = os.path.join(head, "config.yaml")

        args = yaml.load(open(config_path, "r"), yaml.Loader)
        return args