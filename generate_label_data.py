import argparse
from tqdm import tqdm

import numpy as np
import torch

from models.msnet_harmonic_loss import MSNet
from models.ftanet_harmonic_loss import FTANet
from utils import *

def generate_label_data(manifest_path, ckpt, device, batch_size=16, threshold_melody=0.0040, threshold_non_melody=0.0034):
    model = MSNet(device=device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    data_list = load_manifest(manifest_path)
    for data in tqdm(data_list):
        cfp_path = data['cfp_path']
        f0_path = data['f0_path']
        f0ref, f0mask = [], []

        x_data = np.load(cfp_path)
        y_data = np.zeros(x_data.shape[2])
        y_mask = np.zeros(x_data.shape[2])
        x_data, y_data, y_mask = segment_one_piece(x_data, y_data, y_mask)
        
        x_tensor, y_tensor = torch.from_numpy(x_data).float(), torch.from_numpy(y_data)
        for start in range(0, len(x_data), batch_size):
            pred_seg = model((
                x_tensor[start: start + batch_size],
                y_tensor[start: start + batch_size],
                y_tensor[start: start + batch_size],
                y_tensor[start: start + batch_size],
                y_tensor[start: start + batch_size],
            ), requires_loss=False).cpu().numpy()
            
            prob, bin_seg = pred_seg.max(axis=1), pred_seg.argmax(axis=1)
            mask_seg = np.zeros_like(prob)
            mask_seg[(prob > threshold_melody) | (prob < threshold_non_melody)] = 1.0
            bin_seg[prob < threshold_melody] = -1
            f0_seg= bin2f0(bin_seg)

            mask_seg = mask_seg.flatten()
            f0_seg = f0_seg.flatten()
            
            f0ref.extend(f0_seg)
            f0mask.extend(mask_seg)

        lines = [['{:.3f}'.format(i / 100), '{:.3f}'.format(f0ref[i]), '{:.1f}'.format(f0mask[i])] for i in range(len(f0ref))]
        lines = ['\t'.join(line) + '\n' for line in lines]
        
        with open(f0_path, 'w') as f:
            f.writelines(lines)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--manifest_path', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--device', type=str)

    args = parser.parse_args()

    generate_label_data(
        manifest_path=args.manifest_path,
        ckpt=args.ckpt,
        device=args.device
    )