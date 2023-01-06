import argparse
from tqdm import tqdm

import numpy as np
import torch

from models.msnet_harmonic_loss import MSNet
from models.ftanet_harmonic_loss import FTANet
from utils import *

def generate_label_data(manifest_path, ckpt, device, batch_size=16, threshold_melody=0.004, threshold_non_melody=0.0035):
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
            harmonic_fail = (melody_prob > threshold_melody) & (harmonic_prob > threshold_melody) & ((harmonic_bin - melody_bin > 120) | (harmonic_bin - melody_bin < 0))
            subharmonic_fail = (melody_prob > threshold_melody) & (subharmonic_prob > threshold_melody) & ((melody_bin - subharmonic_bin > 120) | (melody_bin - subharmonic_bin < 0))
            
            mask_seg[melody_fail | harmonic_fail | subharmonic_fail] = 0.0
            
            f0ref.extend(f0_seg.flatten())
            f0mask.extend(mask_seg.flatten())
            

        lines = [['{:.3f}'.format(i / 100), '{:.3f}'.format(f0ref[i]), '{:.1f}'.format(f0mask[i])] for i in range(len(f0ref))]
        lines = ['\t'.join(line) + '\n' for line in lines]
        
        with open(f0_path, 'w') as f:
            f.writelines(lines)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--manifest_path', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('-p_min', '--threshold_melody', type=float, default=0.005)
    parser.add_argument('-p_max', '--threshold_non_melody', type=float, default=0.0036)

    args = parser.parse_args()

    generate_label_data(
        manifest_path=args.manifest_path,
        ckpt=args.ckpt,
        device=args.device,
        threshold_melody=args.threshold_melody,
        threshold_non_melody=args.threshold_non_melody
    )