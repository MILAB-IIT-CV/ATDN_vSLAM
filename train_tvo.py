from datetime import datetime

# Third party imports
from tqdm import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard.writer import SummaryWriter
import torch.utils.tensorboard

# Project module imports
from atdn_vslam.odometry.loss import CLVO_Loss
from atdn_vslam.odometry.datasets import FlowKittiDataset3
from atdn_vslam.odometry.tvo import TVO
from atdn_vslam.utils.helpers import log
from atdn_vslam.utils.arguments import Arguments


def train(
    args, 
    model, 
    dataloader, 
    odometry_loss, 
    optimizer, 
    scheduler, 
    writer, 
    epoch
    ):
    #scaler = GradScaler()
    pbar = tqdm(total=len(dataloader))
    
    for batch, (fl, true_rot, true_tr) in enumerate(dataloader):
        true_rot = true_rot.to(args.device)
        true_tr = true_tr.to(args.device)

        loss = 0

        optimizer.zero_grad()

        split_size = 1
        #flows = torch.split(fl, split_size, dim=0)
        #true_rots = torch.split(true_rot, split_size, dim=0)
        #true_trs = torch.split(true_tr, split_size, dim=0)
        #running_loss = 0
        #for flow, rot, tr in zip(flows, true_rots, true_trs):
            #with torch.autocast_mode.autocast(device_type="cuda", dtype=torch.float16):
        #input_data = flow.to(args.device).float()
        
        input_data = fl.to(args.device).float()
        pred_rots, pred_trs = model(input_data)
        loss = odometry_loss(pred_rots, pred_trs, true_rot, true_tr, device=args.device)
        running_loss = loss.item()

        loss.backward()
        #loss.backward(retain_graph=True)

        optimizer.step()
        scheduler.step()
        #scaler.update()
        
        writer.add_scalar('Loss', running_loss, batch+epoch*len(dataloader))
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], batch+epoch*len(dataloader))
        
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
                            pin_memory=False,
                            #persistent_workers=True,
                            #prefetch_factor=2,
                            drop_last=True)

    model = TVO(args.batch_size, in_channels=2).to(args.device)

    print(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log("Trainable parameters:", trainable_params)

    if args.stage > 1:
        load_path = args.weight_file+str(args.stage-1)+".pth"
        log("Loading weigths from ", load_path)
        model.load_state_dict(torch.load(load_path, map_location=args.device))

    # TODO Change back lr
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, eps=args.epsilon)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler_limit = args.epochs*len(dataloader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                        scheduler_limit, 
                                                        eta_min=1e-9)

    loss = CLVO_Loss(args.alpha, w=args.w)

    now = datetime.now()
    writer = SummaryWriter("log/tensorboard/"+str(now.month)+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'('+str(model.__class__.__name__)+')')

    model.train()
    print("============================================ ", "Training", " =============================================\n")
    for epoch in range(args.epochs):
        print("------------------------------------------ ", "Epoch ", epoch+1, "/", args.epochs, " ------------------------------------------\n")

        train(args,
              model, 
              dataloader, 
              loss, 
              optimizer, 
              scheduler, 
              writer,
              epoch)


        save_path = (args.weight_file+str(args.stage) + str(model.__class__.__name__).lower() + model.suffix + ".pth")
        log("Saving model as ", save_path)
        torch.save(model.state_dict(), save_path)

    writer.flush()
    writer.close()

    log("Training started at: ", now, " and ended at: ", datetime.now(), ". Ellapsed time: ", datetime.now()-now)


# Calling main training method
main()