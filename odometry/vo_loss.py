import torch
from utils.helpers import  log
from utils.helpers import euler2matrix, matrix2euler, transform

class CLVO_Loss():

    def __init__(self, alpha=1, w=3):
        self.rot_weight = (1.0/torch.tensor([0.0175, 0.0031, 0.0027])).unsqueeze(0)
        self.tr_weight = (1.0/torch.tensor([0.0219, 0.0260, 1.0917])).unsqueeze(0)
        
        self.last_com = 0
        self.stage = 0
        self.alpha = alpha
        self.delta = 1
        self.khi = 1
        self.w = w


    def __call__(self, pred_transforms, true_transforms, device='cuda'):

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
        L_rel = self.transform_loss(pred_rot, pred_tr, true_rot, true_tr, device).sum(-1)

        # -------------------
        # Composite pose loss
        # -------------------
        L_com = []
        for i in range(len(pred_rot)):
            L_com.append(self.com_loss([pred_rot[i], pred_tr[i]], [true_rot[i], true_tr[i]], w=w, device=device))
        L_com = torch.stack(L_com, dim=0).sum(-1)
        
        # ---------------
        # Total pose loss
        # ---------------
        L_total = self.alpha*L_rel+(1-self.alpha)*L_com
        return L_total.mean()


    def com_loss(self, pred_transforms, true_transforms, w, device='cuda'):

        # Separating rotations and translations
        pred_rot, pred_tr = pred_transforms
        true_rot, true_tr = true_transforms

        # Converting euler vector and translation vector to homogenous transformation matrix
        pred_homogenous_array = []
        true_homogenous_array = []
        for i in range(len(pred_rot)):
            # Converting predicted
            pred_homogenous_array.append(transform(pred_rot[i], pred_tr[i], device=device))

            # Converting true
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
            pred_comm_rot = matrix2euler(pred_comm[:3, :3])
            pred_comm_tr = pred_comm[:3, -1]

            # Converting back to euler and separaing the matrix
            true_comm_rot = matrix2euler(true_comm[:3, :3])
            true_comm_tr = true_comm[:3, -1]
            
            loss = self.transform_loss(pred_comm_rot, pred_comm_tr, true_comm_rot, true_comm_tr, device=device)
            losses.append(loss)

        loss = torch.stack(losses, dim=0)
        return loss


    def transform_loss(self, pred_rotation, pred_translation, true_rotation, true_translation, device='cuda'):

        diff_rotation = pred_rotation-true_rotation*self.rot_weight
        diff_translation = pred_translation-true_translation*self.tr_weight
        
        # Mean is changed to be calculated in call method
        norm_rotation = torch.linalg.norm(diff_rotation, dim=-1, ord='fro')
        norm_translation = torch.linalg.norm(diff_translation, dim=-1, ord='fro')

        loss = self.delta*norm_translation + self.khi*norm_rotation
        return loss

