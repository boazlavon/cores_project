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
import torch
from itertools import chain
from string import punctuation

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import (AdamW, T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup)
from t5_dataset import CoresDataset
from cores_tokens import CoresDatasetPreProcessor
from pl_bolts.callbacks import PrintTableMetricsCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class T5FineTuner(pl.LightningModule):
  def __init__(self, train_builder, val_builder, init_w, dropout, **kwargs):
    super(T5FineTuner, self).__init__()
    self.save_hyperparameters(kwargs)
    
    self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path, dropout_rate=dropout, cache_dir='./cache')
    self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.tokenizer_name_or_path, cache_dir='./cache')

    logging.info(f"Model Dropout: {self.model.config.dropout_rate}")
    cores_tokens = ['<<', '>>', '[[u]]', '[[t]]', '[[f]]']
    self.tokenizer.add_tokens(cores_tokens)
    self.tokenizer.model_max_length = 128
    self.model.resize_token_embeddings(len(self.tokenizer))
    self.cur_val_loss = []
    self.avg_val_losses = []
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
  
  def validation_step(self, batch, batch_idx):
    loss = self._step(batch)
    self.cur_val_loss.append(loss)
    return {"val_loss": loss}
  
  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    tensorboard_logs = {"val_loss": avg_loss}
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

class MyPrintCallback(PrintTableMetricsCallback):
    def on_validation_end(self, trainer, pl_module):
        self.on_epoch_end(trainer, pl_module)

class LoggingCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        print('\n')
        logger.info(f"***** Epoch End ******")

    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        metrics = trainer.callback_metrics
        pl_module.cur_val_loss = torch.tensor(pl_module.cur_val_loss)
        mean = torch.mean(pl_module.cur_val_loss)
        d = ({ 'val_loss' : float(mean), 'epoch' : trainer.current_epoch, 'steps' : trainer.global_step})
        pl_module.avg_val_losses.append(d)
        pl_module.cur_val_loss = []

        # Log results
        logger.info(str(d))
