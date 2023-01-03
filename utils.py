import numpy as np
import torch
import time

import mir_eval

LEN_SEG = 128

def melody_eval(ref_time, ref_freq, est_time, est_freq):

    output_eval = mir_eval.melody.evaluate(ref_time,ref_freq,est_time,est_freq)
    VR = output_eval['Voicing Recall']*100.0 
    VFA = output_eval['Voicing False Alarm']*100.0
    RPA = output_eval['Raw Pitch Accuracy']*100.0
    RCA = output_eval['Raw Chroma Accuracy']*100.0
    OA = output_eval['Overall Accuracy']*100.0
    eval_arr = np.array([VR, VFA, RPA, RCA, OA])
    return eval_arr

def img2bin(pred, threshold=0.4):
    prob, pred_bin = pred.max(axis=1), pred.argmax(axis=1)
    pred_bin[prob < threshold] = -1
    return pred_bin

def bin2img(y):
    N = y.shape[0]
    img = np.zeros([N, 320, LEN_SEG], dtype=np.int64)
    for i in range(N):
        img[i, y[i], np.arange(LEN_SEG)] = 1
        img[i, :, y[i] == -1] = 0 
    return img

def f02bin(y):
    y = np.copy(y)
    y[y > 0] = np.round(np.log2(y[y > 0] / 31) * 60)
    y -= 1 # convert bin (1-320) to index (0-319), silence frame becomes -1
    return y.astype(np.int64) 

def bin2f0(y):
    y = np.copy(y).astype(np.float32)
    y[y >= 0] = 31. * 2 ** (y[y >= 0] / 60)
    y[y < 0] = 0.
    return y

def f02img(y):
    return bin2img((f02bin(y)))

def img2f0(y, threshold=0.0036):
    return bin2f0(img2bin(y, threshold=threshold))

def load_list(path):
    with open(path, 'r') as f:
        data_list = [line.strip() for line in f.readlines()]    
    return data_list

def load_train_data(path):
    tick = time.time()
    train_list = load_list(path)
    X, y = [], []
    num_seg = 0
    for i in range(len(train_list)):
        print('({:d}/{:d}) Loading data: '.format(i+1, len(train_list)), train_list[i])
        X_data, y_data = load_data(train_list[i])
        y_data[(y_data > 1250) | (y_data < 32)] = 0
        seg = len(X_data)
        num_seg += seg
        X.append(X_data)
        y.append(y_data)
        print('({:d}/{:d})'.format(i+1, len(train_list)), train_list[i], 'loaded: ', '{:2d} segments'.format(seg))
    print("Training data loaded in {:.2f}(s): {:d} segments".format(time.time() - tick, num_seg))
    return np.vstack(X), np.vstack(y)

def load_data(fp):
    '''
    X: (N, C, F, T)
    y: (N, T)
    '''
    X = np.load('data/cfp/' + fp)
    L = X.shape[2]
    num_seg = L // LEN_SEG
    X = np.vstack([X[np.newaxis, :, :, LEN_SEG*i:LEN_SEG*i+LEN_SEG] for i in range(num_seg)])

    f = open('data/f0ref/' + fp.replace('.npy', '.txt'))
    y = []
    for line in f.readlines():
        y.append(float(line.strip().split()[1]))
    num_seg = min(len(y) // LEN_SEG, num_seg) # 防止X比y长
    y = np.vstack([y[LEN_SEG*i:LEN_SEG*i+LEN_SEG] for i in range(num_seg)])
    return X, y
