import torch
from utils.helpers import project_depth, log

class Movement_Depth_Loss():
    def __init__(self, device="cuda:0"):
        self.device = device

    def __call__(self, depth1, depth2, flows, calibs, transforms):
        #assert depth1.shape == depth2.shape
        # TODO Try to use torch functions instead of for loop
        batch, height, width = depth1.shape
        losses = []
        for i in range(batch):
            # Index element of batch
            calib = calibs[i]
            points1 = project_depth(depth1[i], calib, device=self.device).permute(1, 2, 0)
            points2 = project_depth(depth2[i], calib, device=self.device)
            flow = flows[i]
            transform = transforms[i]

            # x & Y index tensors
            index_u = torch.arange(0, width, 1).repeat((height, 1)).to(self.device)
            index_v = torch.arange(0, height, 1).repeat((width, 1)).permute(1, 0).to(self.device)

            # Using optical flow to pair 3D points
            index_u = (index_u + flow[0]).long()
            index_v = (index_v + flow[1]).long()
            # Generating mask for modified indexes out of image range
            mask_u = torch.bitwise_and((index_u < width), (index_u >= 0))
            mask_v = torch.bitwise_and((index_v < height), (index_v >= 0))
            mask = torch.bitwise_and(mask_u, mask_v)
            # If modified index goes out of range, truncate to zero (these values will be masked out)
            index_u = torch.where(mask, index_u, 0)
            index_v = torch.where(mask, index_v, 0)

            # Transforming 3D points of second depth with GT transform
            points2 = torch.cat([points2.view(3, height*width), torch.ones((1, height*width), device=self.device)], dim=0)
            points2 = torch.matmul(transform[:3, :].float(), points2)

            points2 = points2.view(3, height, width).permute(1, 2, 0)
            points2 = points2[index_v, index_u, :]

            diff = torch.norm((points2[mask]-points1[mask]), p="fro", dim=-1)
            
            loss = diff.mean()
            losses.append(loss)
            
        losses = torch.stack(losses, dim=0)

        return losses.mean()