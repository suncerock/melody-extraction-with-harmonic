import os
import json
import h5py
import time

import torch
import torch.nn as nn
import torch.utils.data as Data

from utils import *
from dataset import DatasetWithHarmonic
from models.msnet_harmonic_loss import MSNet
from models.ftanet_harmonic_loss import FTANet

DEBUG = 0

def train(train_manifest, test_manifest, batch_size, num_epoch, lr, step_size, threshold, device, save_path):
    if not DEBUG:
        if os.path.exists(save_path):
            raise Exception("{} already exists!".format(save_path))
        os.mkdir(save_path)

    train_x, train_y = [], []
    for manifest_path in train_manifest:
        x, y = load_data_by_segment(manifest_path, progress_bar=False)
        train_x.append(x)
        train_y.append(y)
    train_x, train_y = np.vstack(train_x), np.vstack(train_y)

    dataset = DatasetWithHarmonic(train_x=train_x, train_y=train_y)
    dataloader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    model = FTANet(device=device)

    best_epoch = {name: 0 for name in test_manifest}
    best_OA = {name: 0 for name in test_manifest}
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

    tick = time.time()
    for epoch in range(num_epoch):
        tick_e = time.time()

        model.train()
        train_loss = 0

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            loss, _ = model(batch)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= step + 1
        scheduler.step()

        print("----------------------")
        print("Epoch={:3d}\tTrain_loss={:6.4f}".format(epoch, train_loss))
        model.eval()
        for manifest_path in test_manifest:
            eval_arr = np.zeros(5, dtype=np.float32)
            with torch.no_grad():
                test_x, test_y = load_data_by_piece(manifest_path)
                for i in range(len(test_x)):
                    x_piece, y_piece = segment_one_piece(test_x[i], test_y[i])
                    x_piece, y_piece = torch.from_numpy(x_piece).float(), torch.from_numpy(y_piece).float()
                    length = x_piece.size(0)
                    est_freq = []
                    for start in range(0, length, batch_size):
                        pred_seg = model((
                            x_piece[start: start + batch_size],
                            y_piece[start: start + batch_size],
                            y_piece[start: start + batch_size],
                            y_piece[start: start + batch_size]
                        ), requires_loss=False)
                        est_freq_seg = img2f0(pred_seg.cpu().numpy(), threshold=threshold).flatten()
                        est_freq.extend(est_freq_seg)
                    est_freq = np.array(est_freq)
                    ref_freq = bin2f0(f02bin(y_piece.numpy().flatten()))
                    time_series = np.arange(len(ref_freq))
                    eval_arr += melody_eval(time_series, ref_freq, time_series, est_freq)
                
                eval_arr /= len(test_x)

                print(manifest_path)
                print("Valid: VR={:.2f}\tVFA={:.2f}\tRPA={:.2f}\tRCA={:.2f}\tOA={:.2f}".format(eval_arr[0], eval_arr[1],
                                                                                            eval_arr[2], eval_arr[3],
                                                                                           eval_arr[4]))
                if eval_arr[-1] > best_OA[manifest_path]:
                    best_OA[manifest_path] = eval_arr[-1]
                    best_epoch[manifest_path] = epoch
                print('Best Epoch: ', best_epoch[manifest_path], ' Best OA: ', best_OA[manifest_path])
        if not DEBUG:
            torch.save(model.state_dict(), os.path.join(save_path, 'Epoch_{:d}.pth'.format(epoch)))
        print("Time: {:5.2f}(Total: {:5.2f})".format(time.time() - tick_e, time.time() - tick))


if __name__ == '__main__':
    with open('config.json') as f:
        config = json.load(f)
    train(**config)