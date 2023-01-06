import argparse
import os
import json

def generate_json_from_dir(root_path, output_json, generate_by='cfp'):
    file_list = [filename.split('.')[0] for filename in os.listdir(os.path.join(root_path, generate_by))]
    
    fout = open(output_json, 'w')

    for file in file_list:
        data = dict(
            wav_path=os.path.join(root_path, 'wav', '{}.wav'.format(file)),
            cfp_path=os.path.join(root_path, 'cfp', '{}.npy'.format(file)),
            f0_path=os.path.join(root_path, 'f0ref', '{}.txt'.format(file))
        )
        json.dump(data, fout)
        fout.write('\n')
        fout.flush()

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--root_path', type=str)
    parser.add_argument('-o', '--output_json', type=str)
    parser.add_argument('-m', '--generate_by', type=str, default='cfp')

    args = parser.parse_args()

    generate_json_from_dir(
        root_path=args.root_path,
        output_json=args.output_json,
        generate_by=args.generate_by
    )
