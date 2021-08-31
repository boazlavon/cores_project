import json
import subprocess
import argparse
import random
import logging
import os
import pickle
import threading
import sys
import re
import json

import torch
import pandas as pd

BERT_BART_EVAL_LOSS_RE = '({\'eval_loss\':.*})'
BERT_BART_TRAINING_LOSS_RE = '({\'loss\':.*})'
T5_EVAL_LOSS_RE = '({\'val_loss\':.*})'

LOSS_RE_DICT = {
                'bert' : {'train' : BERT_BART_TRAINING_LOSS_RE, 'eval' : BERT_BART_EVAL_LOSS_RE},
                'bart' : {'train' : BERT_BART_TRAINING_LOSS_RE, 'eval' : BERT_BART_EVAL_LOSS_RE},
                't5' :   {'eval' : T5_EVAL_LOSS_RE}
}

def generate_seq_jsons(model_type, checkpoints_dir, eval_outputdir):
    files = os.listdir(checkpoints_dir)
    files = [f for f in files if 'seq.out' in f]
    for seq_filename in files:
        seq_path = os.path.join(checkpoints_dir, seq_filename)
        with open(seq_path, 'rb') as seq_file:
            data = seq_file.read().decode('utf-8')

        for loss_type, loss_re in LOSS_RE_DICT[model_type].items():
            entries = re.findall(loss_re, data)
            entries = [ json.loads(entry.replace('\'', '"')) for entry in entries ]
            loss_json_path = os.path.join(eval_outputdir, f'{seq_filename}.{loss_type}_loss.json')
            with open(loss_json_path, 'wb') as loss_json:
                loss_json.write(json.dumps(entries).encode('ascii'))
                print(f'{loss_type}: {seq_path} -> {loss_json_path}')

def generate_graphs():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--model', type=str)
    parser.add_argument('--dropout', type=float)
    args = parser.parse_args(sys.argv[1:])
    config = f'{args.dropout}'

    proj_dir = r'.'
    checkpoints_dir = os.path.join(proj_dir, 'training_results', args.model, config)
    if not os.path.isdir(checkpoints_dir):
        print('Please provide an inference directory: {checkpoints_dir}')
        sys.exit(0)

    eval_dir = os.path.join(proj_dir, 'training_graphs_data', args.model, config)
    try:
        os.system(f'mkdir -p {eval_dir}')
    except:
        pass
    generate_seq_jsons(args.model, checkpoints_dir, eval_dir)

if __name__ == '__main__':
    generate_graphs()
