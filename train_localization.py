from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from torchvision.transforms import Normalize, Compose
import torchvision.transforms.functional as TF
from localization.datasets import ColorDataset, DepthDataset, ColorDepthDataset, DoubleColorDataset
from localization.network import MappingVAE
from torch.utils.tensorboard.writer import SummaryWriter
import torch.utils.tensorboard
from tqdm import tqdm
from utils.arguments import Arguments


def train(
    model, 
    dataloader, 
    input_type, 
    optimizer, 
    scheduler,
    writer = None,
    epoch = None
    ):
    pbar = tqdm(total=len(dataloader))
    # TODO augment with colorjitter
    rgb_normalization = Compose([Normalize(mean=(0.0, 0.0, 0.0), std=(255.0, 255.0, 255.0)),
                             Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    
    #hls_normalization = Normalize(mean=(70.2209, 91.0236, 62.9458), std=(47.7068, 78.5522, 63.5109))
    hls_normalization = Normalize(mean=(70.2209, 62.9458), std=(47.7068, 63.5109))
    depth_normalization = Normalize(mean=(0.4855), std=(0.169))
    
    for batch, x in enumerate(dataloader):
        optimizer.zero_grad()

        x = x.to('cuda')
        _, __, ___, x_pred = model(x)

        x = TF.resize(x, list(x_pred.size()[-2:]))
        x = TF.gaussian_blur(x, [5, 5])
        x = rgb_normalization(x)
        
        loss1 = ((x_pred-x)**2).mean()
        sat_true = (x.amax(dim=1) - x.amin(dim=1))
        sat_pred = (x_pred.amax(dim=1) - x_pred.amin(dim=1))
        loss2 = (sat_true - sat_pred).abs().mean()
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        writer.add_scalar('Loss', loss.item(), batch+epoch*len(dataloader))
        writer.add_scalar('LR', scheduler.get_last_lr()[0], batch+epoch*len(dataloader))

        pbar.set_description("Loss: "+ str(loss.item()))
        pbar.update(1)


def main():
    torch.manual_seed(421681)

    num_epochs = 50
    batch_size = 32
    lr = 1e-2
    end_lr = 1e-6
    wd = 1e-3
    input_type = 'RGB'
    division = 5

    args = Arguments.get_arguments()

    dataset = ColorDataset(args.data_path, sequence='00', hls=False, division=division)    
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True, 
                            drop_last=True,
                            pin_memory=True)

    model = MappingVAE().to('cuda')

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    scheduler_limit = num_epochs*len(dataloader)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 
                                               scheduler_limit, 
                                               eta_min=end_lr)


    if not os.path.exists("log"):
        os.mkdir("log")
        os.mkdir("log/localization")
    if not os.path.exists("log/localization"):
        os.mkdir("log/localization")
        
    now = datetime.now()
    writer = SummaryWriter("log/localization/"+str(now.month)+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'('+input_type+')')

    print("============================================ ", "Training", " =============================================\n")
    for epoch in range(num_epochs):
        print("------------------------------------------ ", "Epoch ", epoch+1, "/", num_epochs, " ------------------------------------------\n")

        train(model, dataloader, input_type, optimizer, scheduler, writer, epoch)

        torch.save(model.state_dict(), "checkpoints/localization_"+input_type+'.pth')


main()