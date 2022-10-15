from argparse import ArgumentError
import sys, os
sys.path.insert(0, os.path.abspath(".."))
import torch


class LogShape(torch.nn.Module):
    def __init__(self, message) -> None:
        super(LogShape, self).__init__()
        self.message = message

    def forward(self, input):
        log(self.message+" shape: ", input.shape)
        return input.shape


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


def log(*messages):
    messages = [str(message) for message in messages]
    message = messages[0]

    for msg in messages[1:]:
        message = message + ' ' + msg

    msg_len = len(message)
    print(msg_len*'-')
    print(message)
    print(msg_len*'-')


def line2matrix(pose):
    matrix = torch.stack([pose[0:4], pose[4:8], pose[8:12]], dim=0)
    matrix = torch.cat([matrix, torch.tensor([[0, 0, 0, 1]])], dim=0)

    return matrix


def matrix2line(matrix):
    # TODO: Test function
    pose = torch.cat([matrix[0], matrix[1], matrix[2]], dim=0)

    return pose


def convert_to_KITTI_format(rotations_array, translations_array):
    raise NotImplementedError()


def euler2matrix(r : torch.Tensor, convention="yxz", device="cuda") -> torch.Tensor:
    c1 = torch.cos(r[0])
    c2 = torch.cos(r[1])
    c3 = torch.cos(r[2])

    s1 = torch.sin(r[0])
    s2 = torch.sin(r[1])
    s3 = torch.sin(r[2])
    
    R = None

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

    if R is None:
        raise ArgumentError(None, "convention" + str(convention) + " is not supported")

    return R


def matrix2euler(R, convention="yxz"):
    alpha, beta, gamma = None, None, None
    if convention == "yxz":
        alpha = torch.atan2(R[0, 2], R[2, 2])
        beta = torch.atan2(-R[1, 2], torch.sqrt(1-R[1, 2]**2))
        gamma = torch.atan2(R[1, 0], R[1, 1])
    elif convention == "yxy":
        alpha = torch.atan2(R[0, 1], R[2, 1])
        beta = torch.atan2(torch.sqrt(1-R[1, 1]**2), R[1, 1])
        gamma = torch.atan2(R[1, 0], -R[1, 2])

    return torch.tensor([alpha, beta, gamma])


def transform(rot : torch.Tensor, tr):
    rot = euler2matrix(rot, device="cpu")
    mat = torch.cat([rot, tr.unsqueeze(1).to('cpu')], dim=1)
    mat = torch.cat([mat, torch.tensor([[0, 0, 0, 1]])], dim=0)

    return mat

# TODO Implement Homogenous Matrix to Euler+Trnaslation conversion


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

