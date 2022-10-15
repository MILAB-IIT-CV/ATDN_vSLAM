# PyTorch imports
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

# Project module imports
from odometry.vo_loss import CLVO_Loss
from odometry.vo_datasets import FlowKittiDataset
from odometry.vo_dataloader import FlowKITTIDataLoader
from odometry.clvo import CLVO
from utils.normalization import TrainableStandardization
from utils.helpers import log
from utils.arguments import Arguments
import numpy as np
import math



def train(args, normalization, model, dataloader, odometry_loss, optimizer, scheduler, writer, epoch, log_vals=[]):
    #im_mean, im_std, flows_mean, flows_std = normalization_cache

    for batch, (fl, true_rot, true_tr) in iter(dataloader):

        optimizer.zero_grad()
        loss = 0
        rots, trs = [], []

        for j in range(args.sequence_length):
            flows, true_rotations, true_translations = fl[:, j], true_rot[:, j], true_tr[:, j]

            flows = flows.squeeze().to(args.device)
            flows = normalization(flows)
            input_data = flows

            true_rotations = true_rotations.squeeze().to(args.device)
            true_translations = true_translations.squeeze().to(args.device)
            
            pred_rotations, pred_translations = model(input_data)

            rots.append(pred_rotations)
            trs.append(pred_translations)

        pred_rots = torch.stack(rots, dim=1)
        pred_trs = torch.stack(trs, dim=1)
        loss = odometry_loss([pred_rots, pred_trs], [true_rot.to(args.device), true_tr.to(args.device)], device=args.device)

        log_vals.append(loss.item())
        loss.backward()
        model.reset_lstm()
        
        optimizer.step()
        scheduler.step()

        writer.add_scalar('0: Loss', loss.item(), batch+epoch*len(dataloader))
        writer.add_scalar('1: Learning Rate', scheduler.get_last_lr()[0], batch+epoch*len(dataloader))
        
        print("", end='\r')
        print("Iteration: %d / %d \t|\t Loss: %5f" % (batch, len(dataloader), loss.item()), '\t|\t Learning rate: ', scheduler.get_last_lr(),  end='\n')


def main():
    log("Odometry training")
    print()

    # Setting random seed
    torch.manual_seed(4265664478)
    
    # Instantiating arguments object for optical flow module
    args = Arguments.get_arguments()
    writer = SummaryWriter("loss_log/tensorboard")

    # Instantiating dataset and dataloader
    dataset = FlowKittiDataset(args.data_path, sequences=args.train_sequences, reverse=False, sequence_length=args.sequence_length)
    #dataloader = FlowKITTIDataLoader(dataset=dataset, batch_size=args.batch_size)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, drop_last=True)

    normalization = TrainableStandardization().eval()
    # normalization.load_state_dict(torch.load("checkpoints/norm.pth"))

    model = CLVO(args.batch_size, in_channels=2).to(args.device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log("Trainable parameters:", trainable_params)
    
    #load_path = args.weight_file+str(args.stage-1)+".pth"
    #log("Loading weigths from ", load_path)
    #model.load_state_dict(torch.load(load_path, map_location=args.device))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, eps=args.epsilon)
    #optimizer = optim.RAdam(model.parameters(), lr=args.lr,  weight_decay=args.wd)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     math.floor(args.epochs*(len(dataset)-args.batch_size)/args.batch_size), 
                                                     eta_min=1e-6)

    loss = CLVO_Loss(args.alpha, w=args.w)


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
              writer,
              epoch,
              log_vals_actual)

        log("Saving loss log")
        log_path = args.log_file+str(args.stage-1)+"_"+str(epoch)+".txt"
        np.savetxt(log_path, np.array(log_vals_actual))
        save_path = (args.weight_file+str(args.stage)+".pth")
        log("Saving model as ", save_path)
        torch.save(model.state_dict(), save_path)


# Calling main training method
main()