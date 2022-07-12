import torch
from helpers import  log
from helpers import euler2matrix, matrix2euler

class CLVO_Loss():

    def __init__(self):
        self.last_com = 0
        self.stage = 0
        self.alpha = 0.3
        self.delta = 1
        self.khi = 100


    def __call__(self, pred_transforms, true_transforms, w=4, device='cuda'):

        # --------------------------------------
        # Dimension checking and data extraction
        # --------------------------------------

        assert len(pred_transforms) == len(true_transforms), "Loss: Length of pred and true are not equal"

        pred_rot, pred_tr = pred_transforms
        true_rot, true_tr = true_transforms

        # -------------------
        # Relative pose loss
        # -------------------

        L_rel = 0
        #for i in range(len(pred_rot)):
        #L_rel = L_rel + self.transform_loss(pred_rot[i], pred_tr[i], true_rot[i], true_tr[i], device)
        L_rel = torch.cumsum(self.transform_loss(pred_rot, pred_tr, true_rot, true_tr, device), -2)[:, -1, -1].mean()
        #log("Rel loss: ", L_rel.shape)

        # -------------------
        # Composite pose loss
        # -------------------

        L_com = []
        for i in range(len(pred_rot)):
            L_com.append(self.com_loss([pred_rot[i], pred_tr[i]], [true_rot[i], true_tr[i]], w=w, device=device))
        #log("Comm, loss", L_com)
        L_com = torch.stack(L_com, dim=0)[:, -1].mean()
        #log("Com loss: ", L_com.shape)
        #L_com = L_com if L_com > self.last_com else 0
        
        #if L_com > self.last_com:
            #self.last_com = L_com #.item()
        #else:
            #self.last_com = L_com #.item()
            #L_com = 0
        #log(L_com)
        # ---------------
        # Total pose loss
        # ---------------

        L_total = self.alpha*L_rel+(1-self.alpha)*L_com
        return L_total.mean()


    def com_loss(self, pred_transforms, true_transforms, w=2, device='cuda'):

        # Separating rotations and translations
        pred_rot, pred_tr = pred_transforms
        true_rot, true_tr = true_transforms

        # Converting euler vector and translation vector to homogenous transformation matrix
        pred_homogenous_array = []
        true_homogenous_array = []
        for i in range(w):
            # Converting predicted
            # Euler to matrix rotation representation
            rot_mat = euler2matrix(pred_rot[i], device=device)
            # Concatenating rotation matrix and translation vector

            mat = torch.cat([rot_mat, pred_tr[i].unsqueeze(1)], dim=1)
            # Adding the extra row for homogenous matrix
            mat = torch.cat([mat, torch.tensor([[0, 0, 0, 1]], device=device)], dim=0)
            # Appending matrix to array
            pred_homogenous_array.append(mat)

            # Converting true
            # Euler to matrix rotation representation
            rot_mat = euler2matrix(true_rot[i], device=device)
            # Concatenating rotation matrix and translation vector
            mat = torch.cat([rot_mat, true_tr[i].unsqueeze(1)], dim=1)
            # Adding the extra row for homogenous matrix
            mat = torch.cat([mat, torch.tensor([[0, 0, 0, 1]], device=device)], dim=0)
            # Appending matrix to array
            true_homogenous_array.append(mat)


        # Creating the combining the transformations to a 
        pred_comm = pred_homogenous_array[0]
        true_comm = true_homogenous_array[0]
        for i in range(1, w):
            pred_comm = torch.matmul(pred_comm, pred_homogenous_array[i])
            true_comm = torch.matmul(true_comm, true_homogenous_array[i])

        # Converting back to euler and separaing the matrix
        pred_comm_rot = matrix2euler(pred_comm[:3, :3])
        pred_comm_tr = pred_comm[:3, -1]

        # Converting back to euler and separaing the matrix
        true_comm_rot = matrix2euler(true_comm[:3, :3])
        true_comm_tr = true_comm[:3, -1]
        
        loss = self.transform_loss(pred_comm_rot, pred_comm_tr, true_comm_rot, true_comm_tr, device=device)

        return loss

    def transform_loss(self, pred_rotation, pred_translation, true_rotation, true_translation, device='cuda'):

        pred_rotation = pred_rotation.to(device)
        pred_translation = pred_translation.to(device)

        true_rotation = true_rotation.to(device)
        true_translation = true_translation.to(device)


        diff_translation = pred_translation-true_translation
        diff_rotation = pred_rotation-true_rotation
        
        # Mean is changed to be calculated in call method
        norm_rotation = torch.cumsum((diff_rotation**2), -1)
        norm_translation = torch.cumsum((diff_translation**2), -1)

        loss = self.delta*norm_translation + self.khi*norm_rotation
        return loss

