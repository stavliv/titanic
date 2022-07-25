import torch

class DataLoader():
    def __init__(self, ds, bs):
        self.ds, self.bs = ds, bs
        self.ds[0] = torch.tensor(self.ds[0])
        self.ds[1] = torch.tensor(self.ds[1])
    def __iter__(self):
        for i in range(0, len(self.ds[0]), self.bs): yield (self.ds[0][i:i+self.bs], self.ds[1][i:i+self.bs])