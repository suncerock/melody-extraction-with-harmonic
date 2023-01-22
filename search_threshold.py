import torch

from models.msnet_harmonic_loss import MSNet
from models.mldrnet_harmonic_loss import MLDRnet
from models.ftanet_harmonic_loss import FTANet
from utils import *

device = 'cuda:1'
test_manifest = ["ADC2004.json"]
threshold_list = [0.515, 0.52, 0.525, 0.53]
# threshold_list = [0.535, 0.54, 0.545, 0.55]

model = FTANet(device=device)
model.load_state_dict(torch.load("work_dir/four_class_FTANet/Epoch_7999.pth"))

for threshold in threshold_list:
    print("---------")
    print("Using threshold = {}".format(threshold))
    model.eval()
    for manifest_path in test_manifest:
        eval_arr = np.zeros(5, dtype=np.float32)
        with torch.no_grad():
            test_x, test_y, test_mask = load_data_by_piece(manifest_path)
            for i in range(len(test_x)):
                x_piece, y_piece, mask_piece = segment_one_piece(test_x[i], test_y[i], test_mask[i])
                x_piece, y_piece, mask_piece = torch.from_numpy(x_piece).float(), torch.from_numpy(y_piece).float(), torch.from_numpy(mask_piece).float()
                length = x_piece.size(0)
                est_freq = []
                for start in range(0, length, 16):
                    output = model.forward_feature_map(x_piece[start: start + 16].to(device))
                    output = torch.softmax(output, dim=1)
                    pred_seg = torch.sigmoid(output[:, 1]).cpu().numpy()
                    pred_bin = img2bin(pred_seg, threshold=threshold)
                    
                    # Harmonic check
                    # harmonic_seg = torch.sigmoid(output[:, 2]).cpu().numpy()
                    # subharmonic_seg = torch.sigmoid(output[:, 3]).cpu().numpy()
                    
                    # pred_prob, _ = pred_seg.max(axis=1), pred_seg.argmax(axis=1)
                    # harmonic_prob, harmonic_bin = harmonic_seg.max(axis=1), harmonic_seg.argmax(axis=1)
                    # subharmonic_prob, subharmonic_bin = subharmonic_seg.max(axis=1), subharmonic_seg.argmax(axis=1)
                    
                    # harmonic_result = img2bin(harmonic_seg, threshold=threshold) - 60
                    # subharmonic_result = img2bin(subharmonic_seg, threshold=threshold) + 60
                    # to_substitute = np.where(harmonic_prob > subharmonic_prob, harmonic_result, subharmonic_result)
                    
                    # substitute_frames = ((pred_prob < 0.6)) & ((harmonic_prob > threshold + 0.1) | (subharmonic_prob > threshold + 0.1))
                    # pred_bin[substitute_frames] = to_substitute[substitute_frames]
                    
                    est_freq_seg = bin2f0(pred_bin).flatten()
                    est_freq.extend(est_freq_seg)
                est_freq = np.array(est_freq)
                ref_freq = bin2f0(f02bin(y_piece.numpy().flatten()))
                time_series = np.arange(len(ref_freq))
                eval_arr += melody_eval(time_series, ref_freq, time_series, est_freq)
                # print(melody_eval(time_series, ref_freq, time_series, est_freq))
            eval_arr /= len(test_x)

            print(manifest_path)
            print("Valid: VR={:.2f}\tVFA={:.2f}\tRPA={:.2f}\tRCA={:.2f}\tOA={:.2f}".format(eval_arr[0], eval_arr[1],
                                                                                                eval_arr[2], eval_arr[3],
                                                                                           eval_arr[4]))