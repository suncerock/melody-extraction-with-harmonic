import json
from tqdm import tqdm

import numpy as np
import torch

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

def img2bin(pred, threshold):
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

def img2f0(y, threshold):
    return bin2f0(img2bin(y, threshold=threshold))

def load_manifest(manifest_path):
    with open(manifest_path, 'r') as f:
        data_list = [json.loads(line) for line in f.readlines()]    
    return data_list

def read_one_manifest(manifest, f_min=32, f_max=1250):
    """
    Read one line of manifest

    Parameter
    ----------
    manifest : dict
        dict containing the cfp path and f0 path
    
    Returns
    ----------
    np.ndarray
        (3, F, T), the cfp spectrogram
    np.ndarray
        (T, ), the f0 annotation
    """
    cfp_path = manifest['cfp_path']
    f0_path = manifest['f0_path']
    data_x = np.load(cfp_path)
    with open(f0_path) as f:
        data_y = [float(line.strip().split()[1]) for line in f.readlines()]
    data_y = np.array(data_y, dtype=np.float32)
    data_y[(data_y < f_min) | (data_y > f_max)] = 0.0

    length = min(data_x.shape[2], data_y.shape[0])
    return data_x[..., :length], data_y[:length]

def load_data_by_piece(manifest_path, progress_bar=False):
    """
    Load a list of data

    Returns
    ----------
    List[np.ndarray]
        a list of ndarray with shape (3, F, T)
    List[np.ndarray]
        a list of ndarray with shape (T, )
    """
    data_list = load_manifest(manifest_path)
    x_data, y_data = [], []
    data_list = data_list if not progress_bar else tqdm(data_list)
    for data in data_list:
        x_single, y_single = read_one_manifest(data)
        y_single[(y_single > 1250) | (y_single < 32)] = 0
        x_data.append(x_single)
        y_data.append(y_single)
    return x_data, y_data

def segment_one_piece(x_data, y_data):
    num_seg = len(y_data) // LEN_SEG
    f_bin = x_data.shape[1]
    
    x_data = x_data[..., :num_seg * LEN_SEG].reshape(3, f_bin, num_seg, LEN_SEG).transpose(2, 0, 1, 3)
    y_data = y_data[:num_seg * LEN_SEG].reshape(num_seg, LEN_SEG)

    return x_data, y_data

def load_data_by_segment(manifest_path, progress_bar=True):
    x_data, y_data = load_data_by_piece(manifest_path, progress_bar=progress_bar)
    x_segment, y_segment = [], []
    for x, y in zip(x_data, y_data):
        x_single, y_single = segment_one_piece(x, y)
        x_segment.append(x_single)
        y_segment.append(y_single)
    return np.vstack(x_segment), np.vstack(y_segment)
