import random

import h5py
import numpy as np
import soundfile as sf
import torch.utils.data as Data

from scripts.extract_cfp import feature_extraction, norm, lognorm
from utils import load_manifest, f02img


class AudioDatasetWithHarmonic(Data.Dataset):
    def __init__(self, manifest_path, sr=8000, hop=80, len_seg=128, f_min=32, f_max=1250):
        self.data_list = []
        for manifest in manifest_path:
            self.data_list.extend(load_manifest(manifest))
        
        self.sr = sr
        self.hop = hop
        self.len_seg = len_seg

        self.wav_path = [data['wav_path'] for data in self.data_list]
        self.f0_path = [data['f0_path'] for data in self.data_list]

        self.f0, self.f0_mask = self._load_f0(self.f0_path, f_min, f_max)
        self.f_harmonic, self.f_subharmonic = self._load_harmonic(self.f0, f_min, f_max)

    def __getitem__(self, index):
        label_start = random.randint(0, len(self.f0[index]) - self.len_seg)
        audio_start = label_start * self.hop
        
        with sf.SoundFile(self.wav_path[index]) as f:
            sr = f.samplerate
            assert sr == self.sr

            len_read = self.len_seg * self.hop + self.hop
            f.seek(audio_start)
            audio = f.read(len_read, dtype='float32')

            cfp = self._compute_cfp(audio).astype(np.float32)

        f_melody = self.f0[index][label_start: label_start + self.len_seg]
        f_harmonic = self.f_harmonic[index][label_start: label_start + self.len_seg]
        f_subharmonic = self.f_subharmonic[index][label_start: label_start + self.len_seg]
        f_mask = self.f0_mask[index][label_start: label_start + self.len_seg]
        assert cfp.shape[-1] == len(f_melody), self.wav_path[index]
        return cfp, f02img(f_melody), f02img(f_harmonic), f02img(f_subharmonic), f_mask

    def __len__(self):
        return len(self.data_list)

    def _compute_cfp(self, audio):
        _, _, _, tfrL0, tfrLF, tfrLQ = feature_extraction(audio, self.sr, Hop=self.hop, StartFreq=31.0, StopFreq=1250.0, NumPerOct=60)
        tfrL0 = norm(lognorm(tfrL0))[np.newaxis,:,:]
        tfrLF = norm(lognorm(tfrLF))[np.newaxis,:,:]
        tfrLQ = norm(lognorm(tfrLQ))[np.newaxis,:,:]
        return np.concatenate((tfrL0,tfrLF,tfrLQ),axis=0)

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

class CFPDatasetWithHarmonic(Data.Dataset):
    def __init__(self, manifest_path, len_seg=128, f_min=32, f_max=1250):
        self.data_list = []
        for manifest in manifest_path:
            self.data_list.extend(load_manifest(manifest))
        
        self.len_seg = len_seg

        self.cfp_path = [data['cfp_path'] for data in self.data_list]
        self.f0_path = [data['f0_path'] for data in self.data_list]

        self.f0, self.f0_mask = self._load_f0(self.f0_path, f_min, f_max)
        self.f_harmonic, self.f_subharmonic = self._load_harmonic(self.f0, f_min, f_max)

    def __getitem__(self, index):
        start = random.randint(0, len(self.f0[index]) - self.len_seg)
        
        with h5py.File(self.cfp_path[index]) as f:
            cfp = np.array(f['data'][..., start: start + self.len_seg])

        f_melody = self.f0[index][start: start + self.len_seg]
        f_harmonic = self.f_harmonic[index][start: start + self.len_seg]
        f_subharmonic = self.f_subharmonic[index][start: start + self.len_seg]
        f_mask = self.f0_mask[index][start: start + self.len_seg]
        assert cfp.shape[-1] == len(f_melody), self.wav_path[index]
        return cfp, f02img(f_melody), f02img(f_harmonic), f02img(f_subharmonic), f_mask

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
