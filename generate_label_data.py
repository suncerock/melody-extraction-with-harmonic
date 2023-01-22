import argparse
from tqdm import tqdm

import h5py
import numpy as np
import torch

from models.msnet_harmonic_loss import MSNet
from models.mldrnet_harmonic_loss import MLDRnet
from models.ftanet_harmonic_loss import FTANet
from utils import *

def generate_harmonic_mask(data, model, threshold_melody, threshold_non_melody, device, batch_size=16):
    cfp_path = data['cfp_path']
    
    f0ref, f0mask = [], []

    with h5py.File(cfp_path) as f:
        x_data = np.array(f['data'], dtype=np.float32)
    y_data = np.zeros(x_data.shape[2])
    y_mask = np.zeros(x_data.shape[2])
    x_data, _, _ = segment_one_piece(x_data, y_data, y_mask)

    x_tensor = torch.from_numpy(x_data).float()
    for start in range(0, len(x_data), batch_size):
        with torch.no_grad():
            output = model.forward_feature_map((x_tensor[start: start + batch_size].to(device)))
            output = torch.softmax(output, dim=1).cpu().numpy()
        
        melody_prob, melody_bin = output[:, 1].max(axis=1), output[:, 1].argmax(axis=1)
        harmonic_prob, harmonic_bin = output[:, 2].max(axis=1), output[:, 2].argmax(axis=1)
        subharmonic_prob, subharmonic_bin = output[:, 3].max(axis=1), output[:, 3].argmax(axis=1)

        mask_seg = np.ones_like(melody_prob)
        melody_bin[melody_prob < threshold_melody] = -1
        f0_seg = bin2f0(melody_bin) 
        
        melody_fail = (melody_prob < threshold_melody) & (melody_prob > threshold_non_melody)
        harmonic_fail = (melody_prob > threshold_melody) & (harmonic_prob > threshold_melody) & ((harmonic_bin - melody_bin > 65) | (harmonic_bin - melody_bin < 55))
        subharmonic_fail = (melody_prob > threshold_melody) & (subharmonic_prob > threshold_melody) & ((melody_bin - subharmonic_bin > 65) | (melody_bin - subharmonic_bin < 55))
        
        mask_seg[melody_fail | harmonic_fail | subharmonic_fail] = 0.0
        
        f0ref.extend(f0_seg.flatten())
        f0mask.extend(mask_seg.flatten())
    return f0ref, f0mask

def generate_dropout_mask(data, model, threshold_melody, threshold_non_melody, device, batch_size=16):
    cfp_path = data['cfp_path']
    
    f0ref, f0mask = [], []
    
    with h5py.File(cfp_path, 'r') as f:
        x_data = np.array(f['data'], dtype=np.float32)
        
    y_data = np.zeros(x_data.shape[2])
    y_mask = np.zeros(x_data.shape[2])
    x_data, _, _ = segment_one_piece(x_data, y_data, y_mask)

    x_tensor = torch.from_numpy(x_data).float()
    x_tensor_drop = torch.nn.functional.dropout(x_tensor, p=0.2)
    
    for start in range(0, len(x_data), batch_size):
        with torch.no_grad():
            output = model.forward_feature_map((x_tensor[start: start + batch_size].to(device)))
            output = torch.softmax(output, dim=1).cpu().numpy()

            output_drop = model.forward_feature_map((x_tensor_drop[start: start + batch_size].to(device)))
            output_drop = torch.softmax(output_drop, dim=1).cpu().numpy()

        melody_prob, melody_bin = output[:, 1].max(axis=1), output[:, 1].argmax(axis=1)
        melody_prob_drop, melody_bin_drop = output_drop[:, 1].max(axis=1), output_drop[:, 1].argmax(axis=1)

        mask_seg = np.ones_like(melody_prob)
        melody_bin[melody_prob < threshold_melody] = -1
        melody_bin_drop[melody_prob < threshold_melody] = -1
        f0_seg = bin2f0(melody_bin)

        # melody_fail = (melody_prob < threshold_melody) & (melody_prob > threshold_non_melody)
        melody_fail = np.zeros_like(melody_prob).astype(bool)
        drop_fail = np.abs(melody_bin - melody_bin_drop) > 1
        # noise_fail = np.zeros_like(melody_prob).astype(np.bool)
        
        mask_seg[melody_fail | drop_fail] = 0.0
        f0ref.extend(f0_seg.flatten())
        f0mask.extend(mask_seg.flatten())
    return f0ref, f0mask

def generate_semitone_mask(data, model, threshold_melody, threshold_non_melody, device, batch_size=16):
    cfp_path = data['cfp_path']
    
    f0ref, f0mask = [], []
    
    with h5py.File(cfp_path, 'r') as f:
        x_data = np.array(f['data'], dtype=np.float32)
    
    x_data_up = np.pad(x_data[:, :319], ((0, 0), (1, 0), (0, 0)))
    x_data_down = np.pad(x_data[:, 1:], ((0, 0), (0, 1), (0, 0)))
    
    y_data = np.zeros(x_data.shape[2])
    y_mask = np.zeros(x_data.shape[2])
    x_data, _, _ = segment_one_piece(x_data, y_data, y_mask)
    x_data_up, _, _ = segment_one_piece(x_data_up, y_data, y_mask)
    x_data_down, _, _ = segment_one_piece(x_data_down, y_data, y_mask)

    x_tensor = torch.from_numpy(x_data).float()
    x_tensor_up = torch.from_numpy(x_data_up).float()
    x_tensor_down = torch.from_numpy(x_data_down).float()
    
    for start in range(0, len(x_data), batch_size):
        with torch.no_grad():
            output = model.forward_feature_map((x_tensor[start: start + batch_size].to(device)))
            output_up = model.forward_feature_map((x_tensor_up[start: start + batch_size].to(device)))
            output_down = model.forward_feature_map((x_tensor_down[start: start + batch_size].to(device)))
            
            output = torch.softmax(output, dim=1).cpu().numpy()
            output_up = torch.softmax(output_up, dim=1).cpu().numpy()
            output_down = torch.softmax(output_down, dim=1).cpu().numpy()
            
        melody_prob, melody_bin = output[:, 1].max(axis=1), output[:, 1].argmax(axis=1)
        melody_prob_up, melody_bin_up = output_up[:, 1].max(axis=1), output_up[:, 1].argmax(axis=1)
        melody_prob_down, melody_bin_down = output_down[:, 1].max(axis=1), output_down[:, 1].argmax(axis=1)
        
        mask_seg = np.ones_like(melody_prob)
        melody_bin[melody_prob < threshold_melody] = -1
        melody_bin_up[melody_prob_up < threshold_melody] = -1
        melody_bin_down[melody_prob_down < threshold_melody] = -1
        f0_seg = bin2f0(melody_bin)
        
        upfail = (melody_bin_up != -1) & (melody_bin != -1) & (np.abs(melody_bin - melody_bin_up) > 5)
        downfail = (melody_bin_down != -1) & (melody_bin != -1) & (np.abs(melody_bin - melody_bin_down) > 5)
        
        mask_seg[upfail | downfail] = 0.0
        f0ref.extend(f0_seg.flatten())
        f0mask.extend(mask_seg.flatten())
    return f0ref, f0mask

def generate_noise_mask(data, model, threshold_melody, threshold_non_melody, device, batch_size=16):
    cfp_path = data['cfp_path']
    
    f0ref, f0mask = [], []

    with h5py.File(cfp_path, 'r') as f:
        x_data = np.array(f['data'], dtype=np.float32)
    y_data = np.zeros(x_data.shape[2])
    y_mask = np.zeros(x_data.shape[2])
    x_data, _, _ = segment_one_piece(x_data, y_data, y_mask)

    x_tensor = torch.from_numpy(x_data).float()
    x_tensor_noise = torch.randn_like(x_tensor) * 0.05 + x_tensor

    for start in range(0, len(x_data), batch_size):
        with torch.no_grad():
            output = model.forward_feature_map((x_tensor[start: start + batch_size].to(device)))
            output = torch.softmax(output, dim=1).cpu().numpy()

            output_noise = model.forward_feature_map((x_tensor_noise[start: start + batch_size].to(device)))
            output_noise = torch.softmax(output_noise, dim=1).cpu().numpy()

        melody_prob, melody_bin = output[:, 1].max(axis=1), output[:, 1].argmax(axis=1)
        melody_prob_noise, melody_bin_noise = output_noise[:, 1].max(axis=1), output_noise[:, 1].argmax(axis=1)

        mask_seg = np.ones_like(melody_prob)
        melody_bin[melody_prob < threshold_melody] = -1
        melody_bin_noise[melody_prob < threshold_melody] = -1
        f0_seg = bin2f0(melody_bin)

        # melody_fail = (melody_prob < threshold_melody) & (melody_prob > threshold_non_melody)
        melody_fail = np.zeros_like(melody_prob).astype(bool)
        # noise_fail = np.abs(melody_bin - melody_bin_noise) > 1
        noise_fail = np.zeros_like(melody_prob).astype(np.bool)
        
        mask_seg[melody_fail | noise_fail] = 0.0
        f0ref.extend(f0_seg.flatten())
        f0mask.extend(mask_seg.flatten())
    return f0ref, f0mask

def generate_label_data(manifest_path, ckpt, device, mode='harmonics', batch_size=16, threshold_melody=0.6, threshold_non_melody=0.505):
    model = FTANet(device=device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    data_list = load_manifest(manifest_path)
    for data in tqdm(data_list):
        f0_path = data['f0_path']
        if mode == 'harmonics':
            f0ref, f0mask = generate_harmonic_mask(data, model, threshold_melody, threshold_non_melody, device=device, batch_size=batch_size)
        elif mode == 'noise':
            f0ref, f0mask = generate_noise_mask(data, model, threshold_melody, threshold_non_melody, device=device, batch_size=batch_size)
        elif mode == 'semitone':
            f0ref, f0mask = generate_semitone_mask(data, model, threshold_melody, threshold_non_melody, device=device, batch_size=batch_size)
        elif mode == 'dropout':
            f0ref, f0mask = generate_dropout_mask(data, model, threshold_melody, threshold_non_melody, device=device, batch_size=batch_size)
        elif mode == 'both':
            f0ref, f0mask_harmonic = generate_harmonic_mask(data, model, threshold_melody, threshold_non_melody, device=device, batch_size=batch_size)
            _, f0mask_noise = generate_noise_mask(data, model, threshold_melody, threshold_non_melody, device=device, batch_size=batch_size)
            f0mask = [h * n for h, n in zip(f0mask_harmonic, f0mask_noise)]
        # print((np.array(f0mask) == 0).sum(), (np.array(f0mask) == 1).sum())
        lines = [['{:.3f}'.format(i / 100), '{:.3f}'.format(f0ref[i]), '{:.1f}'.format(f0mask[i])] for i in range(len(f0ref))]
        lines = ['\t'.join(line) + '\n' for line in lines]
        
        with open(f0_path, 'w') as f:
            f.writelines(lines)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--manifest_path', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('-p_min', '--threshold_melody', type=float, default=0.625)
    parser.add_argument('-p_max', '--threshold_non_melody', type=float, default=0.1)
    parser.add_argument('-m', '--mode', type=str, default='harmonics')

    args = parser.parse_args()

    generate_label_data(
        manifest_path=args.manifest_path,
        ckpt=args.ckpt,
        device=args.device,
        mode=args.mode,
        threshold_melody=args.threshold_melody,
        threshold_non_melody=args.threshold_non_melody
    )