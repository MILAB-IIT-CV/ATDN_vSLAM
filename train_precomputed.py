# PyTorch imports
import torch
from torch import optim
from torchvision.transforms import Resize
from torchvision import transforms

# Project module imports
from odometry.vo_loss import CLVO_Loss
from odometry.vo_datasets import KittiOdometryDataset
from odometry.clvo import CLVO
from helpers import log, Arguments
import numpy as np
import math

# Fixing random seed
torch.manual_seed(4265664478)

# Instantiating arguments object for optical flow module
args = Arguments()

# Instantiating dataset and dataloader
batch_size = args.batch_size
sequence_length = args.sequence_length
dataset = KittiOdometryDataset(args.data_path, "00", precomputed_flow=True, sequence_length=sequence_length)
loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=False)


im_mean = torch.load("normalization_cache/rgb_mean.pth").unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(args.device)
im_std = torch.load("normalization_cache/rgb_std.pth").unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(args.device)
flows_mean = torch.load("normalization_cache/flow_mean.pth").unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(args.device)
flows_std = torch.load("normalization_cache/flow_std.pth").unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(args.device)


model = CLVO(args, precomputed_flows=True, in_channels=8).to(args.device)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

resize = Resize(75)

log("Trainable parameters:", trainable_params)

model.load_state_dict(torch.load("odometry/clvo_final_adam_2.pth", map_location=args.device))
print("RGBs mean: ", im_mean.squeeze())
print("RGBs Std: ", im_std.squeeze())

print("Flows mean: ", flows_mean.squeeze())
print("Flows std: ", flows_std.squeeze())


aug = transforms.Compose([
    transforms.ColorJitter(brightness=0.05, saturation=0.05, hue=0.0001)
])

def train(model, dataset, odometry_loss, optimizer, scheduler, aug, log_vals=[]):
    indexes = torch.randperm(len(dataset))
    for i in range(0, len(dataset)-batch_size, batch_size):
    #for batch, (im1, im2, fl, true_rot, true_tr) in enumerate(dataset):

        optimizer.zero_grad()
        loss = 0
        rots, trs = [], []
        
        im1, im2, fl, true_rot, true_tr = [], [], [], [], []
        for j in range(batch_size):
            img1, img2, flow, true_rotation, true_translation = dataset[int(indexes[i+j])]
            im1.append(img1)
            im2.append(img2)
            fl.append(flow)
            true_rot.append(true_rotation)
            true_tr.append(true_translation)
        im1 = torch.stack(im1, dim=0)
        im2 = torch.stack(im2, dim=0)
        fl = torch.stack(fl, dim=0)
        true_rot = torch.stack(true_rot, dim=0)
        true_tr = torch.stack(true_tr, dim=0)


        for j in range(sequence_length):
            img1, img2, flows, true_rotations, true_translations = im1[:, j], im2[:, j], fl[:, j], true_rot[:, j], true_tr[:, j]
            img1 = img1.squeeze().to(args.device)
            img2 = img2.squeeze().to(args.device)
            
            img1 = aug(img1.byte()).float()
            img2 = aug(img2.byte()).float()

            img1 = (img1-im_mean)/im_std
            img2 = (img2-im_mean)/im_std

            flows = flows.squeeze().to(args.device)
            flows = (flows-flows_mean)/flows_std

            input_data = torch.cat([flows, img1, img2], dim=1)
            #input_data = torch.cat([flows, img1], dim=1)
            true_rotations = true_rotations.squeeze().to(args.device)
            true_translations = true_translations.squeeze().to(args.device)

            pred_rotations, pred_translations = model(input_data)
            #loss = loss + odometry_loss([pred_rotations, pred_translations], [true_rotations, true_translations], device=args.device)
            rots.append(pred_rotations)
            trs.append(pred_translations)

        pred_rots = torch.stack(rots, dim=1)
        pred_trs = torch.stack(trs, dim=1)
        loss = odometry_loss([pred_rots, pred_trs], [true_rot.to(args.device), true_tr.to(args.device)], device=args.device)

        #loss = odometry_loss([pred_rotations, pred_translations], [true_rotations, true_translations], device=args.device)
        #loss.backward()
        #avg_loss_buffer += loss.item()
        #loss = loss/batch_size
        log_vals.append(loss.item())
        loss.backward()
        
        optimizer.step()
        scheduler.step()

        print("", end='\r')
        print("Iteration: %d / %d \t|\t Loss: %5f" % (int(i/batch_size), int(len(dataset)/batch_size), loss.item()), '\t|\t Learning rate: ', scheduler.get_last_lr(),  end='\n')
        #print("Iteration: %d / %d \t|\t Loss: %5f" % (batch, len(dataset), loss.item()), '\t|\t Learning rate: ', scheduler.get_last_lr(),  end='\n')
        #print("Iteration: %d / %d \t|\t Loss: %5f" % (batch, len(dataset), avg_loss), '\t|\t Learning rate: ', scheduler.get_last_lr(),  end='\n')


epochs = args.epochs
log_vals = None

#optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, eps=args.epsilon)
#optimizer = optim.RAdam(model.parameters(), lr=args.lr,  weight_decay=args.wd)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs*len(dataset)/batch_size), eta_min=1e-6)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, math.floor(epochs*(len(dataset)-batch_size)/batch_size), eta_min=1e-6)

loss = CLVO_Loss()


aug = transforms.Compose([
    transforms.ColorJitter(brightness=0.05, saturation=0.05, hue=0.01),
])

model.train()
print(" ============================================ ", "Training", " ============================================\n")
for epoch in range(epochs):
    print("------------------------------------------ ", "Epoch ", epoch+1, "/", epochs, " ------------------------------------------\n")
    log_vals_actual = []
    
    train(model, dataset, loss, optimizer, scheduler, aug, log_vals_actual)
    log("Saving loss log")
    np.savetxt('loss_log/final_adam_3_'+str(epoch)+'.txt', np.array(log_vals_actual))
    log("Saving model as last")
    torch.save(model.state_dict(), "odometry/clvo_final_adam_3.pth")
