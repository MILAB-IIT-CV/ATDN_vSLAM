import torch 
from torch.nn import Parameter

class NormalizationForKITTI():
    def __init__(self, device) -> None:
        self.im_mean = torch.load("utils/rgb_mean.pth").unsqueeze(-1).unsqueeze(-1).unsqueeze(0).unsqueeze(0).to(device)
        self.im_std = torch.load("utils/rgb_std.pth").unsqueeze(-1).unsqueeze(-1).unsqueeze(0).unsqueeze(0).to(device)

        self.flows_mean = torch.load("utils/flow_mean.pth").unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
        self.flows_std = torch.load("utils/flow_std.pth").unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)


    def normalize_rgb(self, rgb):
        normalized_rgb = (rgb-self.im_mean)/self.im_std
        return normalized_rgb


    def normalize_flow(self, flow):
        normalized_flow = (flow-self.flows_mean)/self.flows_std
        return normalized_flow


    def log_params(self):
        print("Flows mean: ", self.flows_mean.squeeze())
        print("Flows std: ", self.flows_std.squeeze())

        print("RGBs mean: ", self.im_mean.squeeze())
        print("RGBs Std: ", self.im_std.squeeze())

    
class TrainableStandardization(torch.nn.Module):
    def __init__(self) -> None:
        super(TrainableStandardization, self).__init__()
        
        self.flows_mean = torch.load("utils/flow_mean.pth").unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        self.flows_std = torch.load("utils/flow_std.pth").unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        
        self.flows_mean = Parameter(self.flows_mean, requires_grad=True)
        self.flows_std = Parameter(self.flows_std, requires_grad=True)


    def forward(self, input):
        result = (input-self.flows_mean)/self.flows_std

        return result


    def log_params(self):
        print("Flows mean: ", self.flows_mean.squeeze())
        print("Flows std: ", self.flows_std.squeeze())