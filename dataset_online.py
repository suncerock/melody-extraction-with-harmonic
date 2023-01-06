import numpy as np
import scipy
import torch
import torch.utils.data as Data

from scripts.extract_cfp import feature_extraction, norm, lognorm
from utils import load_manifest, f02img

# y = np.random.randn(800 * 10)
# sr = 8000
# hop = 80

# Z, time, CenFreq, tfrL0, tfrLF, tfrLQ = feature_extraction(y, sr, Hop=hop, StartFreq=31.0, StopFreq=1250.0, NumPerOct=60)
# Z, time, CenFreq, tfrL0, tfrLF, tfrLQ = feature_extraction(y, sr, Hop=hop, StartFreq=20.0, StopFreq=2048.0, NumPerOct=60)
# tfrL0 = norm(lognorm(tfrL0))[np.newaxis,:,:]
# tfrLF = norm(lognorm(tfrLF))[np.newaxis,:,:]
# tfrLQ = norm(lognorm(tfrLQ))[np.newaxis,:,:]
# W = np.concatenate((tfrL0,tfrLF,tfrLQ),axis=0)

class DatasetWithHarmonic(Data.Dataset):
    def __init__(self, manifest_path, sr=8000, hop=80, len_seg=128, f_min=32, f_max=1250):
        self.data_list = load_manifest(manifest_path)
        
        self.sr = sr
        self.hop = hop
        self.len_seg = len_seg

        self.wav_path = [data['wav_path'] for data in self.data_list]
        self.f0_path = [data['f0_path'] for data in self.data_list]

        self.f0, self.f0_mask = self._load_f0(self.f0_path, f_min, f_max)
        self.f_harmonic, self.f_subharmonic = self._load_harmonic(self.f0, f_min, f_max)

    def __getitem__(self, index):
        return self.wav_path[index], f02img(self.f0[index]), f02img(self.f_harmonic[index]), f02img(self.f_subharmonic[index]), self.f0_mask[index]

    def __len__(self):
        return len(self.data_list)

    def _load_f0(self, f0_path_list, f_min, f_max):
        f0, f0_mask = [], []
        for f0_path in f0_path_list:
            with open(f0_path) as f:
                data_y_mask = [(float(line.strip().split()[1]), float(line.strip().split()[2])) for line in f.readlines()]
                data_y_mask = np.array(data_y_mask)
                data_y, data_mask = data_y_mask[:, 0].astype(np.float32), data_y_mask[:, 1].astype(np.int32)
                data_y[(data_y < f_min) | (data_y > f_max)] = 0.0
            f0.append(data_y)
            f0_mask.append(data_mask)
        return f0, f0_mask

    def _load_harmonic(self, f0, f_min, f_max):
        f_harmonic, f_subharmonic = [], []
        for f0_piece in f0:
            f_harmonic_piece = f0_piece * 2
            f_subharmonic_piece = f0_piece / 2

            f_harmonic_piece[f_harmonic_piece > f_max] = 0
            f_subharmonic_piece[f_subharmonic_piece < f_min] = 0
        
            f_harmonic.append(f_harmonic_piece)
            f_subharmonic.append(f_subharmonic_piece)
        return f_harmonic, f_subharmonic

if __name__ == '__main__':
    dataset = DatasetWithHarmonic(manifest_path='train_small.json')
    a, b, c, d, e = dataset[0]
    print(a)
    print(b.shape, c.shape, d.shape, e.shape)