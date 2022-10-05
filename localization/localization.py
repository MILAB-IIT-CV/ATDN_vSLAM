import torch
from torch import dropout, nn
from torchvision.transforms import Resize
from general_layers.conv import Conv, ResidualConv, TransposedConv, InterleaveUpscaling, ConnectedUpscale
from utils.helpers import log


class MappingVAE(nn.Module):
    def __init__(   self, 
                    in_channels=3, 
                    latent_vector_features=1024, 
                    channels=[32, 128, 256, 256, 256, 256, 256, 256, 256],
                    target_size=(376, 1241),
                    down_layer = Conv,
                    up_layer = InterleaveUpscaling):

        super(MappingVAE, self).__init__()

        self.flattened_shape = 1536

        DownScaler = down_layer
        UpScaler = up_layer

        self.encoder = nn.Sequential(
            Conv(in_channels=in_channels, out_channels=in_channels, kernel_size=[7, 7], padding=[3, 3], activation=nn.Mish),            
        
            DownScaler(in_channels=in_channels, out_channels=channels[0], kernel_size=3, stride=2, padding=0, activation=nn.Mish),
            DownScaler(in_channels=channels[0], out_channels=channels[1], kernel_size=3, stride=2, padding=0, activation=nn.Mish),
            DownScaler(in_channels=channels[1], out_channels=channels[2], kernel_size=3, stride=2, padding=0, activation=nn.Mish),
            DownScaler(in_channels=channels[2], out_channels=channels[3], kernel_size=3, stride=2, padding=0, activation=nn.Mish),
            DownScaler(in_channels=channels[3], out_channels=channels[4], kernel_size=3, stride=2, padding=0, activation=nn.Mish),
            #DownScaler(in_channels=channels[4], out_channels=channels[5], kernel_size=3, stride=2, padding=1, activation=nn.Mish),
            #DownScaler(in_channels=channels[5], out_channels=channels[6], kernel_size=3, stride=3, padding=0, activation=nn.Mish),
            #DownScaler(in_channels=channels[6], out_channels=channels[7], kernel_size=3, stride=2, padding=1, activation=nn.Mish),
            #DownScaler(in_channels=channels[7], out_channels=channels[8], kernel_size=3, stride=2, padding=1, activation=nn.Mish),
            
            #nn.Flatten(),
            #nn.Linear(in_features=self.flattened_shape, out_features=latent_vector_features),
            #nn.PReLU(),
            #nn.Dropout(0.2)
        )

        #self.mean_lin = nn.Linear(in_features=latent_vector_features, out_features=latent_vector_features)
        #self.sigma_lin = nn.Linear(in_features=latent_vector_features, out_features=latent_vector_features)
        self.mean_lin = nn.Conv2d(in_channels=channels[4], out_channels=channels[4], kernel_size=[3, 3], stride=[3,  3])
        self.sigma_lin = nn.Conv2d(in_channels=channels[4], out_channels=channels[4], kernel_size=[3, 3], stride=[3, 3])


        self.decoder = nn.Sequential(
            #nn.Linear(in_features=latent_vector_features, out_features=self.flattened_shape),
            #nn.PReLU(),
            #nn.Dropout(0.2),
            #nn.Unflatten(-1, (256, 1, 6)),
            
            #UpScaler(in_channels=channels[8], out_channels=channels[7], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), activation=nn.Mish),
            #UpScaler(in_channels=channels[7], out_channels=channels[6], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), activation=nn.Mish),
            #UpScaler(in_channels=channels[6], out_channels=channels[5], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), activation=nn.Mish),
            #UpScaler(in_channels=channels[5], out_channels=channels[4], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), activation=nn.Mish),
            UpScaler(in_channels=channels[4], out_channels=channels[3], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), activation=nn.Mish),
            UpScaler(in_channels=channels[3], out_channels=channels[2], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), activation=nn.Mish),
            UpScaler(in_channels=channels[2], out_channels=channels[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), activation=nn.Mish),

            UpScaler(in_channels=channels[1], out_channels=channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), activation=nn.Mish),
            UpScaler(in_channels=channels[0], out_channels=in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), activation=nn.Mish),
            Resize(target_size)
        )


    def forward(self, image, VAE=False):
        encoded = self.encoder(image)

        #log("Encoded shape: ", encoded.shape)
        
        if VAE:
            mu = self.mean_lin(encoded)
            logvar = self.sigma_lin(encoded)
            sigma = torch.exp(0.5*logvar)
            eps = torch.randn_like(sigma)
            latent_vector = mu + sigma*eps
        else:
            latent_vector = self.mean_lin(encoded)
            mu = None
            logvar = None

        #log("Mu shape: ", mu.shape)
        #log("Sigma shape: ", sigma.shape)
        #log("Latent vector shape: ", latent_vector.shape)
        
        #decoded = self.decoder(latent_vector)
        decoded = self.decoder(latent_vector)
        #log("Decoded shape: ", decoded.shape)
        
        return mu, logvar, latent_vector, decoded


    def get_code(self, image):
        code = self.encoder(image)
        return code


    def generate_from_code(self, code):
        return self.decoder(code)


# -------------------------------------------------------------------------------------

class MappingUnet(nn.Module):
    def __init__(   self, 
                    in_channels=3, 
                    latent_vector_features=1024, 
                    channels=[32, 64, 128, 128, 256, 256, 256, 512, 512], 
                    down_layer = Conv,
                    up_layer = ConnectedUpscale) -> None:
        super(MappingUnet, self).__init__()

        DownScale = down_layer
        UpScale = up_layer

        self.start_conv = Conv(in_channels=in_channels, out_channels=channels[0], kernel_size=[7, 7], padding=[3, 3], activation=nn.Mish)        

        i = 0
        self.Down1 = DownScale(in_channels=channels[i], out_channels=channels[i+1],  kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], activation=nn.Mish)
        i = i+1
        self.Down2 = DownScale(in_channels=channels[i], out_channels=channels[i+1],  kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], activation=nn.Mish)
        i = i+1
        self.Down3 = DownScale(in_channels=channels[i], out_channels=channels[i+1],  kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], activation=nn.Mish)
        i = i+1
        self.Down4 = DownScale(in_channels=channels[i], out_channels=channels[i+1],  kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], activation=nn.Mish)
        i = i+1
        self.Down5 = DownScale(in_channels=channels[i], out_channels=channels[i+1],  kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], activation=nn.Mish)
        i = i+1
        self.Down6 = DownScale(in_channels=channels[i], out_channels=channels[i+1],  kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], activation=nn.Mish)
        i = i+1
        self.Down7 = DownScale(in_channels=channels[i], out_channels=channels[i+1],  kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], activation=nn.Mish)
        i = i+1
        self.bottleneck_in = nn.Sequential(
            Conv(in_channels=channels[i], out_channels=channels[i+1], kernel_size=[3, 3], stride=[1, 1], padding=[0, 0], activation=nn.Mish),
            nn.Flatten()
            )

        flattened_size = 4096

        self.bottleneck_mean = nn.Linear(in_features=flattened_size, out_features=1024)
        self.bottleneck_sigma = nn.Linear(in_features=flattened_size, out_features=1024)

        self.upscaler_in = nn.Sequential(
            nn.Linear(in_features=1024, out_features=flattened_size),
            nn.Dropout(),
            nn.Mish(inplace=True), 
            nn.Unflatten(-1, (channels[i], 1, 8)),
            Conv(in_channels=channels[i+1], out_channels=channels[i], kernel_size=[3, 3], stride=[1, 1], padding=[2, 2], activation=nn.Mish)
        )


        self.Up1 = UpScale(in_channels=channels[i], out_channels=channels[i-1],  kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], activation=nn.Mish)
        i -= 1
        self.Up2 = UpScale(in_channels=channels[i], out_channels=channels[i-1],  kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], activation=nn.Mish)
        i -= 1
        self.Up3 = UpScale(in_channels=channels[i], out_channels=channels[i-1],  kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], activation=nn.Mish)
        i -= 1
        self.Up4 = UpScale(in_channels=channels[i], out_channels=channels[i-1],  kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], activation=nn.Mish)
        i -= 1
        self.Up5 = UpScale(in_channels=channels[i], out_channels=channels[i-1],  kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], activation=nn.Mish)
        i -= 1
        self.Up6 = UpScale(in_channels=channels[i], out_channels=channels[i-1],  kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], activation=nn.Mish)
        i -= 1
        #self.Up7 = UpScale(in_channels=channels[i], out_channels=channels[i-1],  kernel_size=[3, 3], stride=[1, 1], padding=[1, 1])
        self.Up7 = InterleaveUpscaling(in_channels=channels[i], out_channels=channels[i-1],  kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], activation=nn.Mish)
        i -= 1
        

        self.final_conv = nn.Conv2d(in_channels=channels[i], out_channels=in_channels, kernel_size=[3, 3], padding=[1, 1])
        self.final_resize = Resize((376, 1241))


    def forward(self, input,  sampled=True):
        x = self.start_conv(input)
        
        shortcut1 = self.Down1(x)
        shortcut2 = self.Down2(shortcut1)        
        shortcut3 = self.Down3(shortcut2)        
        shortcut4 = self.Down4(shortcut3)        
        shortcut5 = self.Down5(shortcut4)        
        shortcut6 = self.Down6(shortcut5)        
        shortcut7 = self.Down7(shortcut6)
        downscaled = self.bottleneck_in(shortcut7)       

        latent_mean = self.bottleneck_mean(downscaled)
        latent_logvar = self.bottleneck_sigma(downscaled)
        latent_sigma = torch.exp(0.5*latent_logvar)

        if sampled:
            latent_eps = torch.randn_like(latent_sigma)
            latent_vector = latent_sigma*latent_eps+latent_mean
        else:
            latent_vector = latent_mean

        x = self.upscaler_in(latent_vector)

        x = self.Up1(x, shortcut7)
        x = self.Up2(x, shortcut6)
        x = self.Up3(x, shortcut5)
        x = self.Up4(x, shortcut4)
        x = self.Up5(x, shortcut3)
        x = self.Up6(x, shortcut2)
        x = self.Up7(x)

        x = self.final_conv(x)
        x = self.final_resize(x)

        return latent_mean, latent_logvar, latent_vector, x



# -------------------------------------------------------------------------------------


class PoseRegressor(nn.Module):
    def __init__(self) -> None:
        super(PoseRegressor, self).__init__()

        self.encoder = nn.Sequential(
            Conv(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=[2, 2], padding=[1, 1]),
            Conv(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=[2, 2], padding=[1, 1]),
            nn.Flatten(),
            nn.Linear(in_features=1536, out_features=1024),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024, out_features=512),
            nn.PReLU()
        )

        self.orientation_MLP = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=128, out_features=64),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=64, out_features=32),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=32, out_features=16),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=16, out_features=3)
        )

        self.position_MLP = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=128, out_features=64),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=64, out_features=32),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=32, out_features=16),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=16, out_features=3)
        )

    def forward(self, input):
        code = self.encoder(input)
        orientation = self.orientation_MLP(code)
        position = self.position_MLP(code)
        
        return orientation, position
