import argparse
import time
import sys
import glob
import os
import json
import time
import logging
import random
import re
import pickle
from itertools import chain
from string import punctuation

#import nltk
#nltk.download('punkt')
#from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import (AdamW, T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup)
from t5_dataset import CoresDataset
from cores_tokens import CoresDatasetPreProcessor

os.environ["PYTHONUNBUFFERED"] = '1'
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class T5FineTuner(pl.LightningModule):
  def __init__(self, train_builder, val_builder, model_type, init_w, **kwargs):
    super(T5FineTuner, self).__init__()
    self.save_hyperparameters(kwargs)
    
    if model_type == 't5':
        self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path, cache_dir='./cache')
        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.tokenizer_name_or_path, cache_dir='./cache')
    elif model_type == 'bart':
        self.model = BartForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path, cache_dir='./cache')
        self.tokenizer = BartTokenizer.from_pretrained(self.hparams.tokenizer_name_or_path, cache_dir='./cache')

    cores_tokens = ['<<', '>>', '[[u]]', '[[t]]', '[[f]]']
    self.tokenizer.add_tokens(cores_tokens)
    self.tokenizer.model_max_length = 128
    self.model.resize_token_embeddings(len(self.tokenizer))

    if init_w:
        print('Init Pre-Trained Model Weights')
        self.model.init_weights()
    
    self.train_builder = train_builder
    self.val_builder   = val_builder
  
  def is_logger(self):
    return True
  
  def forward(
      self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
  ):
    return self.model(
        input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        labels=labels,
    )

  def _step(self, batch):
    labels = batch["target_ids"]
    labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

    outputs = self(
        input_ids=batch["source_ids"],
        attention_mask=batch["source_mask"],
        decoder_attention_mask=batch['target_mask'],
        labels=labels
    )

    loss = outputs[0]

    return loss

  def training_step(self, batch, batch_idx):
    loss = self._step(batch)

    tensorboard_logs = {"train_loss": loss}
    return {"loss": loss, "log": tensorboard_logs}
  
  def training_epoch_end(self, outputs):
    #self.log({"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs})
    pass
    #avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
    #tensorboard_logs = {"avg_train_loss": avg_train_loss}

  def validation_step(self, batch, batch_idx):
    loss = self._step(batch)
    return {"val_loss": loss}
  
  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    tensorboard_logs = {"val_loss": avg_loss}
    #logger.info(str({"outputs": outputs}))
    #logger.info(str({"avg_val_loss": avg_loss}))
    return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

  def configure_optimizers(self):
    "Prepare optimizer and schedule (linear warmup and decay)"

    model = self.model
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": self.hparams.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
    self.opt = optimizer
    return [optimizer]
  
  def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
    optimizer.step()
    optimizer.zero_grad()
    self.lr_scheduler.step()
  
  def get_tqdm_dict(self):
    tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
    return tqdm_dict

  def train_dataloader(self):
    train_dataset = CoresDataset(tokenizer=self.tokenizer, builder=self.train_builder, max_len=128)
    dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
    t_total = (
        (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
        // self.hparams.gradient_accumulation_steps
        * float(self.hparams.num_train_epochs)
    )
    scheduler = get_linear_schedule_with_warmup(
        self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
    )
    self.lr_scheduler = scheduler
    return dataloader

  def val_dataloader(self):
    val_dataset = CoresDataset(tokenizer=self.tokenizer, builder=self.val_builder, max_len=128)
    return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
  def on_validation_end(self, trainer, pl_module):
    logger.info("***** Validation results *****")
    if pl_module.is_logger():
      metrics = trainer.callback_metrics
     # Log results
      for key in sorted(metrics):
        if key not in ["log", "progress_bar"]:
          logger.info("{} = {}\n".format(key, str(metrics[key])))

  def on_test_end(self, trainer, pl_module):
    logger.info("***** Test results *****")

    if pl_module.is_logger():
      metrics = trainer.callback_metrics

      # Log and save results to file
      output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
      with open(output_test_results_file, "w") as writer:
        for key in sorted(metrics):
          if key not in ["log", "progress_bar"]:
            logger.info("{} = {}\n".format(key, str(metrics[key])))
            writer.write("{} = {}\n".format(key, str(metrics[key])))

set_seed(42)
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
    num_train_epochs=2,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)

model_type = sys.argv[1]
if model_type not in ('bart', 't5', 'init_t5', 'init_bart'):
    print('Invalid Model Type')
    sys.exit(0)

MODEL_NAMES = {'t5' : 't5-base', 'bart' : "facebook/bart-large"}
model_name_or_path = MODEL_NAMES[model_type.replace('init_', '')]
tokenizer_name_or_path = MODEL_NAMES[model_type.replace('init_', '')]

data_dir   = os.path.join('.', 'coref_data')
output_dir = os.path.join('.', 'training_results', model_type)
args_dict.update({'data_dir': data_dir, 'output_dir': output_dir, 'num_train_epochs' : 4})
args_dict.update({'model_name_or_path' : model_name_or_path, 'tokenizer_name_or_path' :  tokenizer_name_or_path})

args = argparse.Namespace(**args_dict)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=args.output_dir, monitor="val_loss", mode="min", 
    save_last=True,
    every_n_train_steps=5000,
    filename='model_{epoch:02d}_{val_loss:.2f}'
)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    #early_stop_callback=False,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=True,
    callbacks=[LoggingCallback()],
)

train_builder_path = sys.argv[2]
with open(train_builder_path, 'rb') as f:
    train_builder = pickle.load(f)

val_builder_path = sys.argv[3]
with open(val_builder_path, 'rb') as f:
    val_builder = pickle.load(f)

set_seed(42)
model = T5FineTuner(train_builder, val_builder, model_type, False, **args_dict)
#model.to(device)
print(model)
trainer = pl.Trainer(**train_params)
#print(trainer)
trainer.fit(model)
time = str(int(time.time()))
last_filename = os.path.join(output_dir, f't5_{time}')
model.model.save_pretrained(last_filename)
