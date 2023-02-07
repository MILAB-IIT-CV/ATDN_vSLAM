from datetime import datetime

# PyTorch imports
import torch
from torch import optim
from torch.utils.data import DataLoader

from torch.utils.tensorboard.writer import SummaryWriter
import torch.utils.tensorboard

# Project module imports
from odometry.loss import CLVO_Loss
from odometry.datasets import FlowKittiDataset2, FlowKittiDataset3

from odometry.network import ATDNVO
from odometry.lightvo import LightVO
from odometry.symvo import SymVO
from odometry.spatialvo import SpatialVO
from odometry.rspvo import ResidualSpatialVO
from odometry.lsc_vo import LSCVO

from utils.helpers import log
from utils.arguments import Arguments
import numpy as np
import math
from tqdm import tqdm


def train(args, model, dataloader, odometry_loss, optimizer, scheduler, writer, epoch, log_vals=[]):
    model.train()
    pbar = tqdm(total=len(dataloader))
    for batch, (fl, true_rot, true_tr) in enumerate(dataloader):
        true_rot = true_rot.to(args.device)
        true_tr = true_tr.to(args.device)

        loss = 0
        rots, trs = [], []
        #optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad()
        for j in range(args.sequence_length):
            input_data = fl[:, j].to(args.device).float()

            #mag = (input_data[:, 0]**2)+(input_data[:, 1]**2).sqrt()
            #deg = torch.atan2(input_data[:, 1], input_data[:, 0])
            #polar = torch.stack([mag, deg], dim=1)

            pred_rotations, pred_translations = model(input_data)
            
            rots.append(pred_rotations)
            trs.append(pred_translations)

        pred_rots = torch.stack(rots, dim=1).double()
        pred_trs = torch.stack(trs, dim=1).double()

        loss = odometry_loss(pred_rots, pred_trs, true_rot, true_tr, device=args.device).float()

        log_vals.append(loss.item())

        loss.backward()

        optimizer.step()
        scheduler.step()
        
        model.reset_lstm()
        
        writer.add_scalar('Loss', loss.item(), batch+epoch*len(dataloader))
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], batch+epoch*len(dataloader))
        
        #print("", end='\r')
        #print("Iteration: %d / %d \t|\t Loss: %5f" % (batch, len(dataloader), loss.item()), '\t|\t Learning rate: ', scheduler.get_last_lr(),  end='\n')
        pbar.update(1)


def main():
    log("Odometry training")
    print()

    # Setting random seed
    torch.manual_seed(4265664478)
    
    # Instantiating arguments object for optical flow module
    args = Arguments.get_arguments()
    log("Flow augmentation: ", args.augment_flow)
    log("Data path: ", args.data_path)
    
    # Instantiating dataset and dataloader
    dataset = FlowKittiDataset3(args.data_path, 
                               sequences=args.train_sequences, 
                               augment=args.augment_flow, 
                               sequence_length=args.sequence_length)

    dataloader = DataLoader(dataset=dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=0, 
                            pin_memory=True,
                            #persistent_workers=True,
                            #prefetch_factor=2,
                            drop_last=True)

    # TODO add other model constructors
    #model = ATDNVO(args.batch_size, in_channels=2).to(args.device)
    #model = SymVO(args.batch_size, in_channels=2).to(args.device)
    #model = LightVO(args.batch_size, in_channels=2).to(args.device)
    model = LSCVO(args.batch_size, in_channels=2).to(args.device)

    print(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log("Trainable parameters:", trainable_params)

    if args.stage > 1:
        load_path = args.weight_file+str(args.stage-1)+".pth"
        log("Loading weigths from ", load_path)
        model.load_state_dict(torch.load(load_path, map_location=args.device))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, eps=args.epsilon)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler_limit = args.epochs*len(dataloader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     scheduler_limit, 
                                                     eta_min=1e-9)

    loss = CLVO_Loss(args.alpha, w=args.w)

    # Tensorboard
    now = datetime.now()
    model_code = str(model.__class__.__name__) + model.suffix
    writer = SummaryWriter("log/tensorboard/"+str(now.month)+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'('+model_code+')')

    print("============================================ ", "Training", " =============================================\n")
    for epoch in range(args.epochs):
        print("------------------------------------------ ", "Epoch ", epoch+1, "/", args.epochs, " ------------------------------------------\n")
        log_vals_actual = []

        train(args,
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
        save_path = (args.weight_file+str(args.stage) + model_code.lower() + ".pth")
        log("Saving model as ", save_path)
        torch.save(model.state_dict(), save_path)

    writer.flush()
    writer.close()

    log("Training started at: ", now, " and ended at: ", datetime.now(), ". Ellapsed time: ", datetime.now()-now)


# Calling main training method
main()