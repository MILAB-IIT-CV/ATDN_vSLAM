# PyTorch imports
import torch
from torch import optim
from torchvision.transforms import Resize
from torchvision import transforms

# Project module imports
from odometry.vo_loss import CLVO_Loss
from odometry.vo_datasets import CustomKittiOdometryDataset
from odometry.vo_dataloader import CustomKITTIDataLoader
from odometry.clvo import CLVO
from normalization_cache.normalization import NormalizationForKITTI
from helpers import log, get_normalization_cache
from arguments import Arguments
import numpy as np
import math


def train(args, normalization, model, dataloader, odometry_loss, optimizer, scheduler, aug, log_vals=[]):
    #im_mean, im_std, flows_mean, flows_std = normalization_cache

    for batch, (images, fl, true_rot, true_tr) in iter(dataloader):

        optimizer.zero_grad()
        loss = 0
        rots, trs = [], []

        for j in range(args.sequence_length):
            imgs, flows, true_rotations, true_translations = images[:, j:j+2], fl[:, j], true_rot[:, j], true_tr[:, j]
            imgs = imgs.squeeze().to(args.device)

            imgs = aug(imgs.byte()).float()

            #imgs = (imgs-im_mean)/im_std
            imgs = normalization.normalize_rgb(imgs)

            flows = flows.squeeze().to(args.device)
            #flows = (flows-flows_mean)/flows_std
            flows = normalization.normalize_flow(flows)
            
            input_data = torch.cat([flows, imgs[:, 0], imgs[:, 1]], dim=1)
            #input_data = torch.cat([flows, img1], dim=1)
            #input_data = flows

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
        model.reset_lstm()
        
        optimizer.step()
        scheduler.step()

        print("", end='\r')
        print("Iteration: %d / %d \t|\t Loss: %5f" % (batch, len(dataloader), loss.item()), '\t|\t Learning rate: ', scheduler.get_last_lr(),  end='\n')
        #print("Iteration: %d / %d \t|\t Loss: %5f" % (batch, len(dataset), loss.item()), '\t|\t Learning rate: ', scheduler.get_last_lr(),  end='\n')
        #print("Iteration: %d / %d \t|\t Loss: %5f" % (batch, len(dataset), avg_loss), '\t|\t Learning rate: ', scheduler.get_last_lr(),  end='\n')


def main():
    log("Odometry training")
    print()

    # Setting random seed
    torch.manual_seed(4265664478)
    # Instantiating arguments object for optical flow module
    args = Arguments.get_arguments()

    # Instantiating dataset and dataloader
    dataset = CustomKittiOdometryDataset(args.data_path, sequences=args.train_sequences, precomputed_flow=True, sequence_length=args.sequence_length)
    dataloader = CustomKITTIDataLoader(dataset=dataset, batch_size=args.batch_size)

    #normalization_cache = get_normalization_cache(args)
    normalization = NormalizationForKITTI(args.device)

    model = CLVO(args, precomputed_flows=True, in_channels=8).to(args.device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log("Trainable parameters:", trainable_params)
    
    load_path = args.weight_file+str(args.stage-1)+".pth"
    log("Loading weigths from ", load_path)
    model.load_state_dict(torch.load(load_path, map_location=args.device))

    aug = transforms.Compose([
        transforms.ColorJitter(brightness=0.05, saturation=0.05, hue=0.0001)
    ])

    #optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, eps=args.epsilon)
    #optimizer = optim.RAdam(model.parameters(), lr=args.lr,  weight_decay=args.wd)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     math.floor(args.epochs*(len(dataset)-args.batch_size)/args.batch_size), 
                                                     eta_min=1e-6)

    loss = CLVO_Loss(args.alpha, w=args.w)
    aug = transforms.Compose([
        transforms.ColorJitter(brightness=0.05, 
                               saturation=0.05, 
                               hue=0.01),
    ])

    model.train()
    print(" ============================================ ", "Training", " ============================================\n")
    for epoch in range(args.epochs):
        print("------------------------------------------ ", "Epoch ", epoch+1, "/", args.epochs, " ------------------------------------------\n")
        log_vals_actual = []
        
        train(args,
              normalization,
              model, 
              dataloader, 
              loss, 
              optimizer, 
              scheduler, 
              aug, 
              log_vals_actual)

        log("Saving loss log")
        log_path = args.log_file+str(args.stage-1)+"_"+str(epoch)+".txt"
        np.savetxt(log_path, np.array(log_vals_actual))
        save_path = (args.weight_file+str(args.stage)+".pth")
        log("Saving model as ", save_path)
        torch.save(model.state_dict(), save_path)


# Calling main training method
main()