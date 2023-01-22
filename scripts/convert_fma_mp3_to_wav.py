import argparse
import os
import soundfile as sf
import librosa
from tqdm import tqdm

def convert_fma_mp3_to_wav(root_path, sr=8000):
    mp3_path = os.path.join(root_path, 'mp3')
    wav_path = os.path.join(root_path, 'wav')

    if not os.path.exists(wav_path):
        os.mkdir(wav_path)

    mp3_file_list = os.listdir(mp3_path)
    for mp3_file in tqdm(mp3_file_list):
        try:
            y, _ = librosa.load(os.path.join(mp3_path, mp3_file), sr=sr, mono=True)
            
            wav_file = os.path.join(wav_path, mp3_file.replace('.mp3', '.wav'))
            sf.write(wav_file, y, samplerate=sr)
        except:
            pass
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--root_path', type=str)
    parser.add_argument('-sr', '--sr', type=int)

    args = parser.parse_args()

    convert_fma_mp3_to_wav(args.root_path, args.sr)
