import argparse
import time
import sys
import glob
import os
import json
import time
import random
import re
import pickle
import torch
from itertools import chain
from string import punctuation

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import argparse
import logging

from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import (AdamW, T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup)
from t5_dataset import CoresDataset
from cores_tokens import CoresDatasetPreProcessor
from t5_tuner import T5FineTuner, LoggingCallback, MyPrintCallback
from pl_bolts.callbacks import PrintTableMetricsCallback


os.environ["PYTHONUNBUFFERED"] = '1'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('--model', type=str)
parser.add_argument('--epoch', type=int)
parser.add_argument('--train_builder_path', type=str)
parser.add_argument('--val_builder_path', type=str)
parser.add_argument('--dropout', type=float)
input_args = parser.parse_args(sys.argv[1:])

model_type = input_args.model
if model_type not in ('t5', 'init_t5'):
    print(f'Invalid Model Type: {model_type}')
    sys.exit(0)

DEFAULT_DROPOUT = {'t5' : 0.1, 'init_t5' : 0.1}
dropout = DEFAULT_DROPOUT[model_type]
if input_args.dropout and input_args.dropout < 1 and input_args.dropout > 0:
    dropout = input_args.dropout

train_builder_path = input_args.train_builder_path
with open(train_builder_path, 'rb') as f:
    train_builder = pickle.load(f)

val_builder_path = input_args.val_builder_path
with open(val_builder_path, 'rb') as f:
    val_builder = pickle.load(f)

train_epoch=8
if input_args.epoch:
    train_epoch=input_args.epoch

MODEL_NAMES = {'t5' : 't5-base', 'init_t5' : 't5-base'}
model_name_or_path = MODEL_NAMES[model_type]
tokenizer_name_or_path = MODEL_NAMES[model_type]

proj_dir = r'.'
data_dir   = os.path.join('.', 'coref_data')
config = f'{dropout}'
output_dir = os.path.join(proj_dir, 'training_results', f'{model_type}', config)

args_dict = dict(
    data_dir="", # path for data files
    output_dir="", # path to save the checkpoints
    model_name_or_path='',
    tokenizer_name_or_path='',
    max_seq_length=128,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=8,
    eval_batch_size=8,
    num_train_epochs=8,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)
args_dict.update({'data_dir': data_dir, 'output_dir': output_dir, 'num_train_epochs' : train_epoch})
args_dict.update({'model_name_or_path' : model_name_or_path, 'tokenizer_name_or_path' :  tokenizer_name_or_path})
args = argparse.Namespace(**args_dict)

checkpoints_dir = os.path.join(args.output_dir, 'checkpoints')
cp_cb = pl.callbacks.ModelCheckpoint(dirpath=args.output_dir, filename='t5-{epoch:02d}', save_last=True)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    #early_stop_callback=False,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    #checkpoint_callback=True,
    callbacks=[LoggingCallback(), cp_cb],
    auto_scale_batch_size="binsearch",
    auto_lr_find=True,
    logger=True,
    log_every_n_steps=50,
    val_check_interval=0.2
)

set_seed(42)
init_w = 'init' in model_type
model = T5FineTuner(train_builder, val_builder, init_w, dropout, **args_dict)
print(model)
trainer = pl.Trainer(**train_params)
trainer.fit(model)

time = str(int(time.time()))
last_filename = os.path.join(output_dir, f'checkpoint_{model_type}_{time}')
model.model.save_pretrained(last_filename)
