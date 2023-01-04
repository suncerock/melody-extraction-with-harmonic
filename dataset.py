import torch
import torch.utils.data as Data

from utils import f02img
    
class DatasetWithHarmonic(Data.Dataset):
    def __init__(self, train_x, train_y, train_mask):
        self.data_tensor = torch.from_numpy(train_x).float()
        self.mask_tensor = torch.from_numpy(train_mask).int()
        
        self.target_freq = train_y
        
        self.harmonic_freq = train_y * 2
        self.harmonic_freq[self.harmonic_freq > 1250] = 0
        
        self.subharmonic_freq = train_y / 2
        self.subharmonic_freq[self.subharmonic_freq < 32] = 0
        
        self.melody_target = torch.from_numpy(f02img(self.target_freq))
        self.harmonic_target = torch.from_numpy(f02img(self.harmonic_freq))
        self.subharmonic_target = torch.from_numpy(f02img(self.subharmonic_freq))

    def __getitem__(self, index):
        return self.data_tensor[index], self.melody_target[index], self.harmonic_target[index], self.subharmonic_target[index], self.mask_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)