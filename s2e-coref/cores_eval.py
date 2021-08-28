import json
import random
import logging
import os
import pickle
import time, threading, sys

import datasets
from metrics import CorefEvaluator, MentionEvaluator
from datasets import Dataset, concatenate_datasets
from datasets import Dataset, load_metric
from utils import extract_mentions_to_predicted_clusters_from_clusters
from cores_tokens import encode
from consts import SPEAKER_START, SPEAKER_END, NULL_ID_FOR_COREF
from cores_tokens_test import CoresDatasetPreProcessorTest

import torch
import pandas as pd

# Training imports
from transformers import AutoConfig, AutoTokenizer, CONFIG_MAPPING, LongformerConfig, BertConfig, BertTokenizer, BertTokenizerFast
from transformers import BertTokenizerFast
from transformers import T5Tokenizer
from transformers import BertGenerationConfig, BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, EncoderDecoderConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from utils import flatten_list_of_lists

STARTING_TOKEN = '<<'
ENDING_TOKEN = '>>'
UNK_CLUSTER_TOKEN = '[[u]]'
IN_CLUSTER_TOKEN = '[[t]]'
NOT_IN_CLUSTER_TOKEN = '[[f]]'

logger = logging.getLogger(__name__)

def load_pickles():
    os.environ["PYTHONUNBUFFERED"] = '1'
    dataset_builder_path = sys.argv[1]
    print(f'Builder path: {dataset_builder_path}')
    if not os.path.exists(dataset_builder_path):
        print(f'Please generate builder (cores_tokens_test.py script): {dataset_builder_path}')
        sys.exit(0)

    print("Loading Builder")
    with open(dataset_builder_path, 'rb') as f:
        builder = pickle.load(f)

    infer_dir = sys.argv[2]
    if not os.path.isdir(infer_dir):
        print('Please provide an inference directory')
    return builder, infer_dir

def eval():
    logging.basicConfig(level=logging.INFO)
    builder, infer_dir = load_pickles()
    official = True
    if len(sys.argv) > 1:
        if sys.argv[1] == 'f':
          official=False
            
    builder.evaluate(infer_dir, official=official)

if __name__ == '__main__':
    eval()
