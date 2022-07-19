import sys, os
sys.path.insert(0, os.path.abspath(".."))
import torch
import yaml


class GMA_Parameters():
    def __init__(self):

        # GMA parameters
        self.model = "GMA/checkpoints/gma-kitti.pth"
        self.dataset = "kitti"
        self.iters = 12
        self.num_heads = 1
        self.position_only = False
        self.position_and_content = False
        self.mixed_precision = True
        self.replace = False
        self.no_alpha = False
        self.no_residuals = False
        self.model_name = self.model
        self.path = "imgs"       
        
        self.dictionary = {
                           "model" : self.model,
                         "dataset" : self.dataset, 
                           "iters" : self.iters, 
                       "num_heads" : self.num_heads, 
                   "position_only" : self.position_only, 
            "position_and_content" : self.position_and_content, 
                 "mixed_precision" : self.mixed_precision,
                         "replace" : self.replace,
                        "no_alpha" : self.no_alpha,
                    "no_residuals" : self.no_residuals,
                            "imgs" : self.path,
                      "model_name" : self.model
        }

    def __contains__(self, key):
        return key in self.dictionary


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
        self.load_file : str
        self.save_file : str
        self.weight_decay : bool        

    @classmethod
    def get_arguments(cls):
        args = yaml.load(open("config.yaml", "r"), yaml.Loader)
        return args


def log(*messages):

    messages = [str(message) for message in messages]
    message = messages[0]

    for msg in messages[1:]:
        message = message + ' ' + msg

    msg_len = len(message)
    print(msg_len*'-')
    print(message)
    print(msg_len*'-')


class LossLogger():
    def __init__(self, data_num) -> None:
        self.data_num = data_num
        self.count = 0

    def log_loss(self, loss):
        pass


class BetaScheduler():
    def __init__(self, num_iters, warmup_rate=0.7) -> None:
        self.beta = 0
        self.num_iters = num_iters
        self.inc = (1)/(num_iters*warmup_rate)

    def step(self):
        next_beta = self.beta+self.inc
        if  next_beta > 1:
            next_beta = 1

        self.beta = next_beta
        
        return self.beta

    def reset(self):
        self.beta = 0

    def get(self):
        return self.beta



def euler2matrix(r, convention="yxz", device="cuda"):

    c1 = torch.cos(r[0])
    c2 = torch.cos(r[1])
    c3 = torch.cos(r[2])

    s1 = torch.sin(r[0])
    s2 = torch.sin(r[1])
    s3 = torch.sin(r[2])
  
    if convention == "yxz":
        R = torch.tensor([  [c1*c3 + s1*s2*s3, c3*s1*s2 - c1*s3, c2*s1],
                            [c2*s3, c2*c3, -s2],
                            [c1*s2*s3 - c3*s1, c1*c3*s2+s1*s3, c1*c2]], device=device)
    elif convention == "xyx":
        R = torch.tensor([  [c2, s2*s3, c3*s2],
                            [s1*s2, c1*c3-c2*s1*s3, -c3*s3-c2*c3*s1],
                            [-c1*s2, c3*s1+c1*c2*s3, c1*c2*c3-s1*s3]], device=device)
    elif convention == "yxy":
        R = torch.tensor([  [c1*c3-c2*s1*s3, s2*s1, c2*s1*s1+c1*s3],
                            [s2*s3, c2, -s2*c3],
                            [-c3*s1-c2*c1*s3, s2*c1, c2*c1*c3-s1*s3]], device=device)

    return R


def matrix2euler(R, convention="yxz"):
  
    if convention == "yxz":
        alpha = torch.atan2(R[0, 2], R[2, 2])
        beta = torch.atan2(-R[1, 2], torch.sqrt(1-R[1, 2]**2))
        gamma = torch.atan2(R[1, 0], R[1, 1])
    elif convention == "yxy":
        alpha = torch.atan2(R[0, 1], R[2, 1])
        beta = torch.atan2(torch.sqrt(1-R[1, 1]**2), R[1, 1])
        gamma = torch.atan2(R[1, 0], -R[1, 2])

    return torch.tensor([alpha, beta, gamma])


def transform(rot, tr):
    rot = euler2matrix(rot, device="cpu")
    mat = torch.cat([rot, tr.unsqueeze(1).to('cpu')], dim=1)
    mat = torch.cat([mat, torch.tensor([[0, 0, 0, 1]])], dim=0)

    return mat


def rel2abs(rotations, translations):
    homogenous = []

    instance_num = len(rotations)
    for i in range(instance_num):
        homogenous.append(transform(rotations[i], translations[i]))

    global_scale = []
    global_scale.append(homogenous[0])
    for i in range(1, instance_num):
        global_scale.append(torch.matmul(global_scale[i-1], homogenous[i]))
        
    global_scale = torch.stack(global_scale, dim=0)
    global_pos = global_scale[:, :3, -1]
    
    return global_pos
