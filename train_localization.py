from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from torchvision.transforms import Normalize, GaussianBlur
from localization.datasets import ColorDataset, DepthDataset, ColorDepthDataset, DoubleColorDataset
from localization.localization import MappingVAE
from torch.utils.tensorboard.writer import SummaryWriter
import torch.utils.tensorboard
from tqdm import tqdm
from utils.arguments import Arguments


def proximity_aware_reconstruction_loss(image, depth, reconstruction):
    pass


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
    loss_acc = 0.0
    
    rgb_normalization = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    #hls_normalization = Normalize(mean=(70.2209, 91.0236, 62.9458), std=(47.7068, 78.5522, 63.5109))
    hls_normalization = Normalize(mean=(70.2209, 62.9458), std=(47.7068, 63.5109))
    depth_normalization = Normalize(mean=(0.4855), std=(0.169))
    #blur = GaussianBlur(kernel_size=3)
    
    for batch, x in enumerate(dataloader):
        optimizer.zero_grad()

        if input_type == "RGB":
            x = x.to('cuda')
            _, __, ___, x_pred = model(x)

            x = x/255.0
            loss = ((x_pred-x)**2).mean((0, 2, 3)).sum()
            if loss > 1.0:
                loss = ((x_pred-x).abs()).mean((0, 2, 3)).sum()

        elif input_type == 'HLS':
            rgb, hls = x
            rgb = rgb.to('cuda')
            hls = hls.to('cuda')
            x = torch.stack([hls[:, 0], hls[:, 2]], dim=1)
            _, __, ___, y_pred = model(hls)

            y_true = rgb/255.0
            # Calculate loss
            loss = (((y_pred-y_true)**2)).mean((0, 2, 3)).sum()
            if loss > 1.0:
                loss = (((y_pred-y_true)).abs()).mean((0, 2, 3)).sum()

        elif input_type == 'Depth':
            x = x.to('cuda')
            _, __, ___, x_pred = model(x)
            
            x = depth_normalization(x)
            loss = ((x_pred-x)**2).mean((0, 2, 3)).sum()
            if loss > 1.0:
                loss = ((x_pred-x).abs()).mean((0, 2, 3)).sum()
        
        elif input_type == 'CD':
            depth, image = x
            input_x = torch.cat([image, depth], dim=1)
            _, __, ___, x_pred = model(input_x)

            image = hls_normalization(image)
            loss = (image-x_pred)**2
            if loss > 1.0:
                loss = (image-x_pred).abs()
            loss = (depth*loss).mean((0, 2, 3)).sum()

        else:
            raise ValueError("Unkown input type: "+input_type)
        
        #print("Mem: ", torch.cuda.memory_reserved()/1024/1024/1024, "GB")
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_acc += loss.item()
        
        writer.add_scalar('Loss', loss.item(), batch+epoch*len(dataloader))
        writer.add_scalar('LR', scheduler.get_last_lr()[0], batch+epoch*len(dataloader))

        #pbar.set_description("Mem: "+ str(torch.cuda.memory_reserved()/1024/1024/1024)+ "GB")
        pbar.set_description("Loss: "+ str(loss.item()))
        pbar.update(1)


def main():
    torch.manual_seed(421681)

    num_epochs = 50
    batch_size = 64
    lr = 1e-2
    end_lr = 1e-6
    wd = 1e-3
    input_type = 'RGB'
    division = 5

    args = Arguments.get_arguments()

    if input_type == 'RGB':
        dataset = ColorDataset(args.data_path, sequence='00', hls=False, division=division)
    elif input_type == 'HLS':
        #dataset = ColorDataset(args.data_path, sequence='00', hls=True, division=division)
        dataset = DoubleColorDataset(args.data_path, sequence='00', division=division)
    elif input_type == 'Depth':
        dataset = DepthDataset(args.data_path, sequence='00', division=division)
    elif input_type == 'CD':
        dataset = ColorDepthDataset(args.data_path, sequence='00', hls=True, division=division)
    else:
        raise ValueError('Unkown input type')
    
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True, 
                            drop_last=True,
                            pin_memory=True)

    model = MappingVAE(input_type=input_type).to('cuda')
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler_limit = num_epochs*len(dataloader)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 
                                               scheduler_limit, 
                                               eta_min=end_lr)

    now = datetime.now()
    writer = SummaryWriter("log/localization/"+str(now.month)+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'('+input_type+')')

    print("============================================ ", "Training", " =============================================\n")
    for epoch in range(num_epochs):
        print("------------------------------------------ ", "Epoch ", epoch+1, "/", num_epochs, " ------------------------------------------\n")

        train(model, dataloader, input_type, optimizer, scheduler, writer, epoch)

        torch.save(model.state_dict(), "checkpoints/localization_"+input_type+'.pth')


main()