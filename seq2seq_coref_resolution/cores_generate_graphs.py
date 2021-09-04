import json
from matplotlib import pyplot as plt
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

def generate_seq_json(model_type, checkpoints_dir, eval_outputdir):
    files = os.listdir(checkpoints_dir)
    files = [f for f in files if 'seq.out.1' ==  f or 'seq.out' == f]
    files.sort()
    seq_filename = files[0]
    seq_path = os.path.join(checkpoints_dir, seq_filename)
    with open(seq_path, 'rb') as seq_file:
        data = seq_file.read().decode('utf-8')
    #for loss_type, loss_re in LOSS_RE_DICT[model_type].items():
    loss_type = 'eval'
    loss_re = LOSS_RE_DICT[model_type.replace('init_', '')][loss_type]
    entries = re.findall(loss_re, data)
    entries = [ json.loads(entry.replace('\'', '"').replace('"val_loss', '"eval_loss')) for entry in entries ]
    loss_json_path = os.path.join(eval_outputdir, f'{seq_filename}.{loss_type}_loss.json')
    with open(loss_json_path, 'wb') as loss_json:
        loss_json.write(json.dumps(entries).encode('ascii'))
        print(f'{loss_type}: {seq_path} -> {loss_json_path}')
    return entries

def generate_graph(model_type, eval_loss_data, main_eval_dir, strip=0, min_epoch=10):
    if strip:
        fig_path = os.path.join(main_eval_dir, f'eval_loss_{model_type}_striped.jpeg')
    else:
        fig_path = os.path.join(main_eval_dir, f'eval_loss_{model_type}.jpeg')
    fig = plt.figure(figsize=(8,4), dpi=300)
    graph_data = {}
    min_length = float('inf')
    min_epoch = float('inf')
    for config, data in eval_loss_data.items():
        if not data:
            continue
        min_length = min(len(data), min_length)

    for config, data in eval_loss_data.items():
        if not data:
            continue
        start = int(strip * min_length)
        xy = [ (i['epoch'], i['eval_loss']) for i in data[start:min_length] ]
        xy = [ (epoch, eval_loss) for (epoch, eval_loss) in xy if epoch < min_epoch ]
        graph_data[config] = {
                                'x' : [ epoch for (epoch, eval_loss) in xy ], 
                                'y' : [ eval_loss for (epoch, eval_loss) in xy ]
                             }
        #graph_data[config] = {
        #                        'x' : [i['epoch']     for i in data[start:min_length]], 
        #                        'y' : [i['eval_loss'] for i in data[start:min_length]], 
        #                     }
        print(f'{config}')
        print(f'x length: {len(graph_data[config]["x"])}')
        print(f'y length: {len(graph_data[config]["y"])}')

    fig = plt.figure()
    fig.suptitle(f'Training {model_type}')
    plt.xlabel('epoch')
    plt.ylabel('eval loss')
    for config, data in eval_loss_data.items():
        if not data:
            continue
        plt.plot(graph_data[config]['x'], graph_data[config]['y'], '--.', label=f'dropout: {config}')

    plt.legend(numpoints=1)
    plt.savefig(fig_path)
    print(f'Save fig: {fig_path}')

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--model', type=str)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--strip', type=float, default=0.15)
    args = parser.parse_args(sys.argv[1:])

    if args.strip > 1 or args.strip < 0:
        print(f'Invalid strip {args.strip}')
        sys.exit(0)

    proj_dir = r'.'
    main_checkpoints_dir = os.path.join(proj_dir, 'training_results', args.model)
    if not os.path.isdir(main_checkpoints_dir):
        print('Please provide an inference directory: {checkpoints_dir}')
        sys.exit(0)

    main_eval_dir = os.path.join(proj_dir, 'training_graphs_data', args.model)
    try:
        os.system(f'mkdir -p {main_eval_dir}')
    except:
        pass
    eval_loss_data = {}
    for config in os.listdir(main_checkpoints_dir):
        checkpoints_dir = os.path.join(main_checkpoints_dir, config)
        eval_dir = os.path.join(main_eval_dir, config)
        try:
            os.system(f'mkdir -p {eval_dir}')
        except:
            pass
        eval_loss_data[config] = generate_seq_json(args.model, checkpoints_dir, eval_dir)
    generate_graph(args.model ,eval_loss_data, main_eval_dir, min_epoch=args.epoch)
    plt.figure().clear()
    generate_graph(args.model ,eval_loss_data, main_eval_dir, strip=args.strip, min_epoch=args.epoch)
if __name__ == '__main__':
    main()
