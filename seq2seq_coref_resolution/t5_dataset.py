import argparse
import glob
import os
import json
import time
import logging
import random
import re
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

from transformers import (AdamW, T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup)
from cores_tokens import CoresDatasetPreProcessor

class CoresDataset(Dataset):
    def __init__(self, tokenizer, builder, max_len):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.builder = builder
        self.inputs = []
        self.targets = []
        self._build()
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask    = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze() # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
    
    def _build(self):
        for i in range(len(self.builder.mentions_df)):
            input_str  = self.builder.mentions_df['input_str'][i]
            output_str = self.builder.mentions_df['output_str'][i]
            
             # tokenize inputs
            tokenized_inputs = self.tokenizer([input_str],  padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
             # tokenize targets
            tokenized_targets = self.tokenizer([output_str],  padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)

        for i in range(len(self.builder.clusters_df)):
            input_str  = self.builder.clusters_df['input_str'][i]
            output_str = self.builder.clusters_df['output_str'][i]
            
             # tokenize inputs
            tokenized_inputs = self.tokenizer([input_str],  padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")

             # tokenize targets
            tokenized_targets = self.tokenizer([output_str],  padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)
