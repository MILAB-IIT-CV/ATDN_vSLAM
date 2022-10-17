import torch 
from torch.nn import Parameter

class Standardization():
    def __init__(self, device) -> None:
        self.im_mean = torch.load("utils/rgb_mean.pth", map_location=device).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        self.im_std = torch.load("utils/rgb_std.pth", map_location=device).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).unsqueeze(0)

        self.flows_mean = torch.load("utils/flow_mean.pth", map_location=device).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        self.flows_std = torch.load("utils/flow_std.pth", map_location=device).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)


    def standardize_rgb(self, rgb):
        normalized_rgb = (rgb-self.im_mean)/self.im_std
        return normalized_rgb


    def standardize_flow(self, flow):
        normalized_flow = (flow-self.flows_mean)/self.flows_std
        return normalized_flow


    def log_params(self):
        print("Flows mean: ", self.flows_mean.squeeze())
        print("Flows std: ", self.flows_std.squeeze())

        print("RGBs mean: ", self.im_mean.squeeze())
        print("RGBs Std: ", self.im_std.squeeze())

    
class FlowStandardization(torch.nn.Module):
    def __init__(self, device="cuda:0") -> None:
        super(FlowStandardization, self).__init__()
        
        # TODO Try changing multiple unsqueeze() to single view()
        self.flows_mean = torch.load("utils/flow_mean.pth", map_location=device).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        self.flows_std = torch.load("utils/flow_std.pth", map_location=device).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)


    def forward(self, input):
        result = (input-self.flows_mean)/self.flows_std
        return result


    def log_params(self):
        print("Flows mean: ", self.flows_mean.squeeze())
        print("Flows std: ", self.flows_std.squeeze())