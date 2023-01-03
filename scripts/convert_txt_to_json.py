import argparse
import json
import os

def convert_txt_to_json(txt_path, root_path, output_json):
    """
    Convert a txt file to json file containing the path to the cfp data and f0 annotation
    """
    with open(txt_path, 'r') as f:
        file_list = [line.strip().replace('.npy', '') for line in f.readlines()]
    
    fout = open(output_json, 'w')

    for file in file_list:
        data = dict(
            cfp_path=os.path.join(root_path, 'cfp', '{}.npy'.format(file)),
            f0_path=os.path.join(root_path, 'f0ref', '{}.txt'.format(file))
        )
        json.dump(data, fout)
        fout.write('\n')
        fout.flush()

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--txt_path', type=str)
    parser.add_argument('-d', '--root_path', type=str)
    parser.add_argument('-o', '--output_json', type=str)

    args = parser.parse_args()

    convert_txt_to_json(
        txt_path=args.txt_path,
        root_path=args.root_path,
        output_json=args.output_json
    )
