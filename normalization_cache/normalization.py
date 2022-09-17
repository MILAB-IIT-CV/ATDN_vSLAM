import torch 


class NormalizationForKITTI():
    def __init__(self, device) -> None:
        self.im_mean = torch.load("normalization_cache/rgb_mean.pth").unsqueeze(-1).unsqueeze(-1).unsqueeze(0).unsqueeze(0).to(device)
        self.im_std = torch.load("normalization_cache/rgb_std.pth").unsqueeze(-1).unsqueeze(-1).unsqueeze(0).unsqueeze(0).to(device)
        print("RGBs mean: ", self.im_mean.squeeze())
        print("RGBs Std: ", self.im_std.squeeze())

        self.flows_mean = torch.load("normalization_cache/flow_mean.pth").unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
        self.flows_std = torch.load("normalization_cache/flow_std.pth").unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
        print("Flows mean: ", self.flows_mean.squeeze())
        print("Flows std: ", self.flows_std.squeeze())


    def normalize_rgb(self, rgb):
        normalized_rgb = (rgb-self.im_mean)/self.im_std
        return normalized_rgb

    def normalize_flow(self, flow):
        normalized_flow = (flow-self.flows_mean)/self.flows_std
        return normalized_flow