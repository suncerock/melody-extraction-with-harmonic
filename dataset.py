import torch
import torch.utils.data as Data

from utils import f02img

class Dataset(Data.Dataset):
    def __init__(self, data_tensor, target_freq):
        self.data_tensor = torch.from_numpy(data_tensor).float()
        self.target_freq = target_freq
        self.target_tensor = torch.from_numpy(f02img(target_freq))

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)
    
class DatasetWithHarmonic(Data.Dataset):
    def __init__(self, data_tensor, target_freq):
        self.data_tensor = torch.from_numpy(data_tensor).float()
        
        self.target_freq = target_freq
        
        self.harmonic_freq = target_freq * 2
        self.harmonic_freq[self.harmonic_freq > 1250] = 0
        
        self.subharmonic_freq = target_freq / 2
        self.subharmonic_freq[self.subharmonic_freq < 32] = 0
        
        self.melody_target = torch.from_numpy(f02img(self.target_freq))
        self.harmonic_target = torch.from_numpy(f02img(self.harmonic_freq))
        self.subharmonic_target = torch.from_numpy(f02img(self.subharmonic_freq))

    def __getitem__(self, index):
        return self.data_tensor[index], self.melody_target[index], self.harmonic_target[index], self.subharmonic_target[index]

    def __len__(self):
        return self.data_tensor.size(0)