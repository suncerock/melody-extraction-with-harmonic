import argparse
import os
import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm

def convert_fma_mp3_to_wav(root_path):
    mp3_path = os.path.join(root_path, 'mp3')
    wav_path = os.path.join(root_path, 'wav')

    if not os.path.exists(wav_path):
        os.mkdir(wav_path)

    mp3_file_list = os.listdir(mp3_path)
    for mp3_file in tqdm(mp3_file_list):
        mp3 = AudioSegment.from_mp3(os.path.join(mp3_path, mp3_file))
        
        wav_file = os.path.join(wav_path, mp3_file.replace('.mp3', '.wav'))
        mp3.export(wav_file, format="wav")
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--root_path', type=str)

    args = parser.parse_args()

    convert_fma_mp3_to_wav(args.root_path)
