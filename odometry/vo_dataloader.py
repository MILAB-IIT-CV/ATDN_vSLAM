import torch
import time
from helpers import log

class CustomKITTIDataLoader():
    def __init__(self, dataset, batch_size) -> None:
        self.batch_size = batch_size
        self.dataset = dataset
        self.indexes = torch.randperm(len(self.dataset))


    def __iter__(self):
        self.indexes_index = 0        
        return self


    def __next__(self):
        if (self.indexes_index+self.batch_size) >= len(self.indexes):
            raise StopIteration

        images, fl, true_rot, true_tr = [], [], [], []

        #actual_time = time.time()

        for j in range(self.batch_size):
            imgs, flow, true_rotation, true_translation = self.dataset[int(self.indexes[self.indexes_index+j])]
            #print("Dataset lookup: ", (time.time()-actual_time)*1000)
            #actual_time = time.time()

            images.append(imgs)
            fl.append(flow)
            true_rot.append(true_rotation)
            true_tr.append(true_translation)

            #print("Append: ", (time.time()-actual_time)*1000)
            #actual_time = time.time()

        #actual_time = time.time()
        #print()

        images = torch.stack(images, dim=0)
        fl = torch.stack(fl, dim=0)
        true_rot = torch.stack(true_rot, dim=0)
        true_tr = torch.stack(true_tr, dim=0)

        #print("Stack: ", (time.time()-actual_time)*1000)
        #actual_time = time.time()

        self.indexes_index = self.indexes_index+self.batch_size
        return int(self.indexes_index/self.batch_size), (images, fl, true_rot, true_tr)


    def __len__(self):
        return int(len(self.dataset)/self.batch_size)


class FlowKITTIDataLoader():
    def __init__(self, dataset, batch_size) -> None:
        self.batch_size = batch_size
        self.dataset = dataset
        self.indexes = torch.randperm(len(self.dataset))


    def __iter__(self):
        self.indexes_index = 0        
        return self


    def __next__(self):
        if (self.indexes_index+self.batch_size) >= len(self.indexes):
            raise StopIteration

        fl, true_rot, true_tr = [], [], []

        #actual_time = time.time()

        for j in range(self.batch_size):
            flow, true_rotation, true_translation = self.dataset[int(self.indexes[self.indexes_index+j])]

            fl.append(flow)
            true_rot.append(true_rotation)
            true_tr.append(true_translation)

        #actual_time = time.time()
        #print()

        fl = torch.stack(fl, dim=0)
        true_rot = torch.stack(true_rot, dim=0)
        true_tr = torch.stack(true_tr, dim=0)

        #print("Stack: ", (time.time()-actual_time)*1000)
        #actual_time = time.time()

        self.indexes_index = self.indexes_index+self.batch_size
        return int(self.indexes_index/self.batch_size), (fl, true_rot, true_tr)


    def __len__(self):
        return int(len(self.dataset)/self.batch_size)