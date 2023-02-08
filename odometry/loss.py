import torch
from torch import nn
from utils.helpers import  log
from utils.transforms import matrix2euler, transform


class CLVO_Loss():

    def __init__(
        self, 
        alpha=1, 
        w=3, 
        device="cuda:0"
    ):
        self.rot_weight = (1.0/torch.tensor([0.0175, 0.0031, 0.0027], device=device)).unsqueeze(0)
        self.tr_weight  = (1.0/torch.tensor([0.0219, 0.0260, 1.0917], device=device)).unsqueeze(0)
        
        self.last_com = 0
        self.alpha = alpha
        self.delta = 1
        self.khi = 100
        self.w = w


    def __call__(
        self, 
        pred_rot, 
        pred_tr, 
        true_rot, 
        true_tr, 
        device='cuda:0'
    ):


        w = self.w

        # -------------------
        # Relative pose loss
        # -------------------
        L_rel = self.transform_loss(pred_rot, pred_tr, true_rot, true_tr).sum(-1)

        # -------------------
        # Composite pose loss
        # -------------------
        com_loss = 0
        #com_loss = []
        #for i in range(len(pred_rot)):
        #    com_loss.append(self.com_loss(pred_rot[i], pred_tr[i], true_rot[i], true_tr[i], w=w, device=device))
        #com_loss = torch.stack(com_loss, dim=0).sum(-1)
        
        # ---------------
        # Total pose loss
        # ---------------
        L_total = self.alpha*L_rel+(1-self.alpha)*com_loss
        return L_total.mean()


    def com_loss(
        self, 
        pred_rot, 
        pred_tr, 
        true_rot, 
        true_tr, 
        w, 
        device='cuda'
    ):

        # Converting euler vector and translation vector to homogenous transformation matrix
        pred_homogenous_array = []
        true_homogenous_array = []
        for i in range(len(pred_rot)):
            pred_homogenous_array.append(transform(pred_rot[i], pred_tr[i], device=device))
            true_homogenous_array.append(transform(true_rot[i], true_tr[i], device=device))

        losses = []
        for j in range(len(pred_homogenous_array)-w):
            # Creating the combining the transformations to a 
            pred_comm = pred_homogenous_array[j]
            true_comm = true_homogenous_array[j]
            for i in range(j+1, j+w+1):
                pred_comm = torch.matmul(pred_comm, pred_homogenous_array[i])
                true_comm = torch.matmul(true_comm, true_homogenous_array[i])
            
            # Converting back to euler and separaing the matrix
            pred_comm_rot = matrix2euler(pred_comm[:3, :3], device=device)
            pred_comm_tr = pred_comm[:3, -1]

            # Converting back to euler and separaing the matrix
            true_comm_rot = matrix2euler(true_comm[:3, :3], device=device)
            true_comm_tr = true_comm[:3, -1]
            
            loss = self.transform_loss(pred_comm_rot, pred_comm_tr, true_comm_rot, true_comm_tr)
            losses.append(loss)

        loss = torch.stack(losses, dim=0)
        return loss


    def transform_loss(
        self, 
        pred_rotation, 
        pred_translation, 
        true_rotation, 
        true_translation
    ):

        diff_rotation = (pred_rotation-true_rotation)
        diff_translation = (pred_translation-true_translation)
        
        norm_rotation = (diff_rotation**2).sum(dim=-1)
        norm_translation = (diff_translation**2).sum(dim=-1)

        loss = (self.delta*norm_translation) + (self.khi*norm_rotation)
        return loss

