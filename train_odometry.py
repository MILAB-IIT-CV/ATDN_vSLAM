from datetime import datetime

# PyTorch imports
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torch.utils.tensorboard

# Project module imports
from odometry.vo_loss import CLVO_Loss
from odometry.vo_datasets import FlowKittiDataset
from odometry.clvo import CLVO
from utils.helpers import log
from utils.arguments import Arguments
import numpy as np
import math
from tqdm import tqdm


def train(args, model, dataloader, odometry_loss, optimizer, scheduler, writer, epoch, log_vals=[]):

    for batch, (fl, true_rot, true_tr) in tqdm(enumerate(dataloader)):

        optimizer.zero_grad()
        loss = 0
        rots, trs = [], []

        for j in range(args.sequence_length):
            flows = fl[:, j]

            flows = flows.squeeze().to(args.device)
            input_data = flows

            pred_rotations, pred_translations = model(input_data)

            rots.append(pred_rotations)
            trs.append(pred_translations)

        pred_rots = torch.stack(rots, dim=1)
        pred_trs = torch.stack(trs, dim=1)
        true_rot = true_rot.to(args.device)
        true_tr = true_tr.to(args.device)
        
        loss = odometry_loss([pred_rots, pred_trs], [true_rot, true_tr], device=args.device)

        log_vals.append(loss.item())
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        model.reset_lstm()

        writer.add_scalar('Loss', loss.item(), batch+epoch*len(dataloader))
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], batch+epoch*len(dataloader))
        
        #print("", end='\r')
        #print("Iteration: %d / %d \t|\t Loss: %5f" % (batch, len(dataloader), loss.item()), '\t|\t Learning rate: ', scheduler.get_last_lr(),  end='\n')


def main():
    log("Odometry training")
    print()

    # Setting random seed
    torch.manual_seed(4265664478)
    
    # Instantiating arguments object for optical flow module
    args = Arguments.get_arguments()
    log("Flow augmentation: ", args.augment_flow)
    # Instantiating dataset and dataloader
    dataset = FlowKittiDataset(args.data_path, sequences=args.train_sequences, augment=args.augment_flow, sequence_length=args.sequence_length)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    model = CLVO(args.batch_size, in_channels=2).to(args.device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log("Trainable parameters:", trainable_params)

    now = datetime.now()
    writer = SummaryWriter("results/tensorboard/"+str(now)[:10]+str(now.hour)+str(now.minute))

    if args.stage > 1:
        load_path = args.weight_file+str(args.stage-1)+".pth"
        log("Loading weigths from ", load_path)
        model.load_state_dict(torch.load(load_path, map_location=args.device))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, eps=args.epsilon)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)


    scheduler_limit = args.epochs*math.ceil((len(dataset)-args.batch_size)/args.batch_size)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     scheduler_limit, 
                                                     eta_min=1e-9)

    loss = CLVO_Loss(args.alpha, w=args.w)


    model.train()
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
        save_path = (args.weight_file+str(args.stage)+".pth")
        log("Saving model as ", save_path)
        torch.save(model.state_dict(), save_path)

    writer.flush()
    writer.close()

    log("Training started at: ", now, " and ended at: ", datetime.now(), ". Ellapsed time: ", datetime.now()-now)


# Calling main training method
main()