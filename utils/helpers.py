import sys, os
sys.path.insert(0, os.path.abspath(".."))
import numpy as np
import torch


class ShapeLogLayer(torch.nn.Module):
    """
    Torch module (layer) for seamlessly logging output shapes in a torch.nn.Sequential model

    :param message: Message to print before output shape. Useful to differentiate between prints.
    """
    def __init__(self, message : str = '') -> None:
        super(ShapeLogLayer, self).__init__()
        self.message = message

    def forward(self, input):
        log(self.message+" shape: ", input.shape)
        return input


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
    """
    Visually decorate prints

    :param messages: Undefined number of message arguments separated with spaces in the log. Any type of message that can be converted with str() is accepted.
    """
    messages = [str(message) for message in messages]
    message = messages[0]

    for msg in messages[1:]:
        message = message + ' ' + msg

    msg_len = len(message)
    print(msg_len*'-')
    print(message)
    print(msg_len*'-')

