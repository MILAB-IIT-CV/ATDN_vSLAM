from torchvision.transforms import Compose, Normalize


def get_rgb_norm():
    return Compose([Normalize(mean=(0.0, 0.0, 0.0), std=(255.0, 255.0, 255.0)),
                    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

def get_flow_norm():
        #Normalize(mean=[0.0, 0.0], std=[41.2430, 41.1322])
        return Normalize(mean=[0.0, 0.0], std=[58.1837, 17.7647])