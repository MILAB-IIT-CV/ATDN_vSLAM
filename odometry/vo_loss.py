import torch
from utils.helpers import  log
from utils.transforms import matrix2euler, transform


class CLVO_Loss():

    def __init__(self, alpha=1, w=3, device="cuda:0"):
        self.rot_weight = (1.0/torch.tensor([0.0175, 0.0031, 0.0027], device=device)).unsqueeze(0)
        self.tr_weight  = (1.0/torch.tensor([0.0219, 0.0260, 1.0917], device=device)).unsqueeze(0)
        
        self.last_com = 0
        self.alpha = alpha
        self.delta = 1
        self.khi = 100
        self.w = w


    def __call__(self, pred_transforms, true_transforms, device='cuda:0'):

        # --------------------------------------
        # Dimension checking and data extraction
        # --------------------------------------
        w = self.w
        assert len(pred_transforms) == len(true_transforms), "Loss: Length of pred and true are not equal"

        pred_rot, pred_tr = pred_transforms
        true_rot, true_tr = true_transforms

        # -------------------
        # Relative pose loss
        # -------------------
        L_rel = 0
        L_rel = self.transform_loss1(pred_rot, pred_tr, true_rot, true_tr).sum(-1)

        # -------------------
        # Composite pose loss
        # -------------------
        com_loss = []
        for i in range(len(pred_rot)):
            com_loss.append(self.com_loss([pred_rot[i], pred_tr[i]], [true_rot[i], true_tr[i]], w=w, device=device))
        com_loss = torch.stack(com_loss, dim=0).sum(-1)
        
        # ---------------
        # Total pose loss
        # ---------------
        L_total = self.alpha*L_rel+(1-self.alpha)*com_loss
        return L_total.mean()


    def com_loss(self, pred_transforms, true_transforms, w, device='cuda'):

        # Separating rotations and translations
        pred_rot, pred_tr = pred_transforms
        true_rot, true_tr = true_transforms

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
            
            loss = self.transform_loss1(pred_comm_rot, pred_comm_tr, true_comm_rot, true_comm_tr)
            losses.append(loss)

        loss = torch.stack(losses, dim=0)
        return loss


    def transform_loss1(self, pred_rotation, pred_translation, true_rotation, true_translation):

        diff_rotation = (pred_rotation-true_rotation)
        diff_translation = (pred_translation-true_translation)
        
        norm_rotation = (diff_rotation**2).sum(dim=-1)
        norm_translation = (diff_translation**2).sum(dim=-1)

        loss = self.delta*norm_translation + self.khi*norm_rotation
        return loss


    def transform_loss2(self, pred_rotation, pred_translation, true_rotation, true_translation):

        diff_rotation = (pred_rotation-true_rotation)
        diff_translation = (pred_translation-true_translation)

        diff_rotation = (diff_rotation*self.rot_weight).squeeze()
        diff_translation = (diff_translation*self.tr_weight).squeeze()
        
        norm_rotation = torch.linalg.vector_norm(diff_rotation, dim=-1, ord=2)
        norm_translation = torch.linalg.vector_norm(diff_translation, dim=-1, ord=2)

        loss = norm_translation + norm_rotation
        return loss

