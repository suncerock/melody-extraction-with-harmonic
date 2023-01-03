import os
import json
import pickle

import torch
import torch.nn as nn
import torch.utils.data as Data

from utils import *
from dataset import Dataset, DatasetWithHarmonic
# from msnet import MSNet
from msnet_harmonic_loss import MSNet


def train(train_list, test_list, batch_size, num_epoch, lr, device, save_path):
    if os.path.exists(save_path):
        raise Exception("{} already exists!".format(save_path))
    os.mkdir(save_path)

    if not os.path.exists('train_data.pkl'):
        X_train, y_train = load_train_data(path=train_list)
        with open('train_data.pkl', 'wb') as f:
            pickle.dump((X_train, y_train), f, protocol=4)
    else:
        with open('train_data.pkl', 'rb') as f:
            X_train, y_train = pickle.load(f)
    test_list = load_list(path=test_list)

    # dataset = Dataset(data_tensor=X_train, target_freq=y_train)
    dataset = DatasetWithHarmonic(data_tensor=X_train, target_freq=y_train)
    dataloader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    model = MSNet(device=device)

    best_epoch = 0
    best_OA = 0
    time_series = np.arange(128) * 0.01
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    tick = time.time()
    for epoch in range(num_epoch):
        tick_e = time.time()

        model.train()
        train_loss = 0

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            loss, pred = model(batch)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= step + 1
        # continue
        model.eval()
        eval_arr = np.zeros(5, dtype=np.float32)
        with torch.no_grad():
            for i in range(len(test_list)):
                X_test, y_test = load_data(test_list[i])
                X_test, y_test = torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
                length_x = X_test.size(0)
                est_freq = []
                for start in range(0, length_x, 8):
                    pred_seg = model((
                        X_test[start:start+8], y_test[start:start+8], y_test[start:start+8], y_test[start:start+8]), requires_loss=False)
                    est_freq_seg = img2f0(pred_seg.cpu().numpy()).flatten()
                    est_freq.extend(est_freq_seg)
                est_freq = np.array(est_freq)
                ref_freq = bin2f0(f02bin(y_test.numpy().flatten()))
                time_series = np.arange(len(ref_freq))
                eval_arr += melody_eval(time_series, ref_freq, time_series, est_freq)
            
            eval_arr /= len(test_list)

            print("----------------------")
            print("Epoch={:3d}\tTrain_loss={:6.4f}".format(epoch, train_loss))
            print("Valid: VR={:.2f}\tVFA={:.2f}\tRPA={:.2f}\tRCA={:.2f}\tOA={:.2f}".format(eval_arr[0], eval_arr[1],
                                                                                        eval_arr[2], eval_arr[3],
                                                                                        eval_arr[4]))
            if eval_arr[-1] > best_OA:
                best_OA = eval_arr[-1]
                best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_path, 'OA_{:.2f}_epoch_{:d}.pth'.format(eval_arr[4], epoch)))
            print('Best Epoch: ', best_epoch, ' Best OA: ', best_OA)
            print("Time: {:5.2f}(Total: {:5.2f})".format(time.time() - tick_e, time.time() - tick))


if __name__ == '__main__':
    with open('config.json') as f:
        config = json.load(f)
    train(**config)