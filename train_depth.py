from datetime import datetime
import math
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import torch.utils.tensorboard

from torch.utils.data import DataLoader
from utils.arguments import Arguments
from utils.helpers import log
from depth.depth_predictor import DepthPredictor
from depth.depth_dataset import OdometryDepthDataset
from depth.depth_loss import Movement_Depth_Loss


def train(model, loss_fn, dataloader, optimizer, scheduler, writer, epoch):
    for batch, (imgs, flows, transforms, calib) in enumerate(dataloader):
        optimizer.zero_grad()
        
        imgs = imgs.to("cuda:0")
        flows = flows.to("cuda:0")
        transforms = transforms.to("cuda:0")
        calib = calib.to("cuda:0")

        input = torch.cat([flows, imgs[:, 0, :, :, :], imgs[:, 1, :, :, :]], dim=1)
        depth1, depth2 = model(input)

        loss = loss_fn(depth1, depth2, flows, calib, transforms)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        writer.add_scalar('Loss', loss.item(), batch+epoch*len(dataloader))
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], batch+epoch*len(dataloader))
        
        print("", end='\r')
        print("Iteration: %d / %d \t|\t Loss: %5f" % (batch, len(dataloader), loss.item()), '\t|\t Learning rate: ', scheduler.get_last_lr(),  end='\n')


def main():
    args = Arguments.get_arguments()

    dataset = OdometryDepthDataset(args.data_path, sequences=['00'], sequence_length=1)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=2)

    model = DepthPredictor().eval()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log("Trainable parameters:", trainable_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler_limit = math.floor(args.epochs*(len(dataset)//2-args.batch_size)/args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     scheduler_limit, 
                                                     eta_min=1e-6)

    loss_fn = Movement_Depth_Loss()
    now = datetime.now()
    writer = SummaryWriter("loss_log/depth/tensorboard/"+str(now)[:10]+str(now.hour)+str(now.minute))
    
    imgs, flow, transforms, calib = dataset[5]
    im1, im2 = imgs[0].unsqueeze(0), imgs[1].unsqueeze(0)
    flow = flow.unsqueeze(0)
    input = torch.cat([flow, im1, im2], dim=1).to("cpu")
    with torch.no_grad():
        writer.add_graph(model, input)
    

    model = model.to("cuda:0").train()
    epochs = 20
    print("============================================ ", "Training", " =============================================\n")
    for epoch in range(epochs):
        print("------------------------------------------ ", "Epoch ", epoch+1, "/", args.epochs, " ------------------------------------------\n")
        
        train(model, loss_fn, dataloader, optimizer, scheduler, writer, epoch)
        log("Saving model to /checkpoints/depth_predictor.pth")
        torch.save(model.state_dict(), "/checkpoints/depth_predictor.pth")

    writer.flush()
    writer.close()

main()