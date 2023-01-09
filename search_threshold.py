import torch

from models.msnet_harmonic_loss import MSNet
from utils import *

device = 'cuda:0'
test_manifest = ["ADC2004.json"]
threshold_list = [0.515, 0.52, 0.525, 0.53]
# threshold_list = [0.535, 0.54, 0.545, 0.55]

model = MSNet(device=device)
model.load_state_dict(torch.load("work_dir/fma4000_MSNet/Epoch_179.pth"))

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
                    pred_seg = model((
                        x_piece[start: start + 16],
                        y_piece[start: start + 16],
                        y_piece[start: start + 16],
                        y_piece[start: start + 16],
                        mask_piece[start: start + 16]
                    ), requires_loss=False)
                    est_freq_seg = img2f0(pred_seg.cpu().numpy(), threshold=threshold).flatten()
                    est_freq.extend(est_freq_seg)
                est_freq = np.array(est_freq)
                ref_freq = bin2f0(f02bin(y_piece.numpy().flatten()))
                time_series = np.arange(len(ref_freq))
                # print(((ref_freq == 0) & (est_freq == 0)).sum() / (est_freq == 0).sum())
                eval_arr += melody_eval(time_series, ref_freq, time_series, est_freq)
            
            eval_arr /= len(test_x)

            print(manifest_path)
            print("Valid: VR={:.2f}\tVFA={:.2f}\tRPA={:.2f}\tRCA={:.2f}\tOA={:.2f}".format(eval_arr[0], eval_arr[1],
                                                                                                eval_arr[2], eval_arr[3],
                                                                                           eval_arr[4]))