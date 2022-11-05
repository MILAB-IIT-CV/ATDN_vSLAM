import torch
from utils.depth import project_depth


class Movement_Depth_Loss():
    def __init__(self, device="cuda:0"):
        self.device = device
        self.sky_depth = 250

    def __call__(self, depth1, depth2, flows, calibs, transforms, masks):
        #assert depth1.shape == depth2.shape
        
        # TODO Try to use torch functions instead of for loop
        batch, height, width = depth1.shape
        losses = []
        for i in range(batch):
            # Index element of batch
            calib = calibs[i]
            flow = flows[i]
            transform = transforms[i]

            L_reconstruction = self.reconstruction_loss(depth1[i], depth2[i], calib, flow, transform, masks[i])
            L_sky1 = self.sky_loss(depth1[i], masks[i, 0])
            L_sky2 = self.sky_loss(depth2[i], masks[i, 1])
            
            #print("Reconstruction loss: ", L_reconstruction.item())
            #print("Sky loss 1: ", L_sky1.item())
            #print("Sky loss 2: ", L_sky2.item())

            loss = L_reconstruction + 0.001*(L_sky1 + L_sky2)

            losses.append(loss)
            
        losses = torch.stack(losses, dim=0)

        return losses.mean()


    def reconstruction_loss(self, depth1, depth2, calib, flow, transform, masks):
            height, width = depth1.shape

            depth_1 = depth1.clone()
            depth_2 = depth2.clone()
            depth_1[masks[0]] = 255
            depth_2[masks[1]] = 255

            points1 = project_depth(depth_1, calib, device=self.device).permute(1, 2, 0)
            points2 = project_depth(depth_2, calib, device=self.device)

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

            return loss


    def sky_loss(self, depth, skymask):
        if skymask.float().sum() > 0.0:
            loss = abs(depth[skymask] - self.sky_depth).mean()
        else:
            loss = 0.0
        return loss