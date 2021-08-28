from transformers import BertTokenizerFast
from transformers import T5Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.models.bart.modeling_bart import shift_tokens_right

from cores_tokens import CoresDatasetPreProcessor
import datasets
from datasets import Dataset, concatenate_datasets
import pickle
import random
import os
import sys
import argparse
import logging

from transformers import AutoConfig, AutoTokenizer, CONFIG_MAPPING, LongformerConfig, BertConfig, BertTokenizer
from transformers import BertGenerationConfig, BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, EncoderDecoderConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainingArguments, Trainer
from transformers import T5ForConditionalGeneration

import numpy as np
import random
import torch

os.environ["PYTHONUNBUFFERED"] = '1'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

batch_size = 8
proj_dir = r'.'
data_dir = os.path.join(proj_dir, 'coref_data')

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('--model', type=str)
parser.add_argument('--epoch', type=int)
parser.add_argument('--dropout', type=float)
args = parser.parse_args(sys.argv[1:])

model_type = args.model
if model_type not in ('bart', 'bert', 'init_bert', 'init_bart'):
    logging.error(f'Invalid Model Type {model_type}')
    sys.exit(0)

DEFAULT_DROPOUT = {'bart' : 0.1, 'init_bart' : 0.1}
dropout = DEFAULT_DROPOUT[model_type] #default bart config
if args.dropout and args.dropout < 1 and args.dropout > 0:
    dropout = args.dropout

init_w = 'init' in model_type
model_type_no_init = model_type.replace('init_', '')
training_dataset_path = os.path.join(data_dir, f'{model_type_no_init}_train_dataset.pkl')
val_dataset_path = os.path.join(data_dir, f'{model_type_no_init}_val_dataset.pkl')
config = f'{dropout}'
checkpoints_dir = os.path.join(proj_dir, 'training_results', f'{model_type}', config)

latest_checkpoint = None
if os.path.isdir(checkpoints_dir):
    checkpoints = os.listdir(checkpoints_dir)
    checkpoints = [ c for c in checkpoints if 'checkpoint' in c ]
    if len(checkpoints):
        checkpoints.sort()
        latest_checkpoint = checkpoints[-1]
        latest_checkpoint = os.path.join(checkpoints_dir, latest_checkpoint)

logging.info(f'Training {model_type} Model!')
logging.info(f'Loading Training Dataset: {training_dataset_path}')
with open(training_dataset_path, 'rb') as f:
    train_df = pickle.load(f)
logging.info(f"Training: {len(train_df)}")

logging.info(f'Loading Validation Dataset: {val_dataset_path}')
with open(val_dataset_path, 'rb') as f:
    val_df = pickle.load(f)
logging.info(f"Validation: {len(val_df)}")

if 'bert' in model_type:
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
if 't5' in model_type:
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
if 'bart' in model_type:
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token
cores_tokens = ['<<', '>>', '[[u]]', '[[t]]', '[[f]]']
tokenizer.add_tokens(cores_tokens)
tokenizer.model_max_length = 128

if latest_checkpoint:
    logging.info(f'Loading latest checkpoint: {latest_checkpoint}')
    if 'bert' in model_type:
        model = EncoderDecoderModel.from_pretrained(latest_checkpoint)
    elif 't5' in model_type:
        model = T5ForConditionalGeneration.from_pretrained(latest_checkpoint)
    elif 'bart' in model_type:
        model = BartForConditionalGeneration.from_pretrained(latest_checkpoint)
else:
    logging.info(f'Building new {model_type} from pre-trained model')
    if 'bert' in model_type:
        encoder = BertGenerationEncoder.from_pretrained("bert-base-uncased")
        decoder = BertGenerationDecoder.from_pretrained("bert-base-uncased", add_cross_attention=True, is_decoder=True)
        encoder.resize_token_embeddings(len(tokenizer))
        decoder.resize_token_embeddings(len(tokenizer))
        if init_w:
            logging.info('Init Pre-Trained Model Weights')
            encoder.init_weights()
            decoder.init_weights()
        model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
        model.config.vocab_size = model.config.decoder.vocab_size
        model.config.decoder_start_token_id = tokenizer.bos_token_id
    elif 't5' in model_type:
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        model.resize_token_embeddings(len(tokenizer))
        if init_w:
            logging.info('Init Pre-Trained Model Weights')
            model.init_weights()
    elif 'bart' in model_type:
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", dropout=dropout)
        model.resize_token_embeddings(len(tokenizer))
        if init_w:
            logging.info('Init Pre-Trained Model Weights')
            model.init_weights()

logging.info(f"Model Dropout: {model.config.dropout}")
#model.config.dropout = dropout
model.config.max_length = 128
model.config.early_stopping = True
model.config.num_beams = 4
model.config.no_repeat_ngram_size = 2
#model.config.length_penalty = 2.0

if False:
    logging_dir=os.path.join(checkpoints_dir, 'logs')
    try:
        os.mkdir(logging_dir)
    except:
        pass

    train_epoch = 4
    if args.epoch:
        train_epoch = args.epoch
    training_args = TrainingArguments(
                        output_dir=checkpoints_dir,          
                        evaluation_strategy="steps",
                        num_train_epochs=train_epoch,
                        per_device_train_batch_size=batch_size, 
                        per_device_eval_batch_size=1,   
                        logging_steps=500,  # set to 1000 for full training
                        save_steps=2500,  # set to 500 for full training
                        warmup_steps=500,               
                        weight_decay=0.01,              
                        logging_dir=logging_dir,
                        eval_steps=5000000,  # set to 8000 for full training
                        save_total_limit=5,
                        overwrite_output_dir=True
                        )
    trainer = Trainer(
                  model=model,                       
                  args=training_args,                  
                  train_dataset=train_df,
                  eval_dataset=val_df,
    )
    trainer.train()
    sys.exit(0)

else: # for BERT
    # Generic configs
    # set special tokens
    # model.config.eos_token_id = tokenizer.eos_token_id
    # model.config.pad_token_id = tokenizer.pad_token_id

    # sensible parameters for beam search
    #model.config.min_length = 40
    #model.config.no_repeat_ngram_size = 3
    #model.config.length_penalty = 2.0

    train_batch_size=batch_size
    val_batch_size=batch_size
    save_steps=500
    if 'bart' in model_type:
        train_batch_size=8
        val_batch_size=2
        save_steps=5000

    logging.info(f'train_batch_size={train_batch_size}, val_batch_size={val_batch_size}')
    # set training arguments - these params are not really tuned, feel free to change
    training_args = Seq2SeqTrainingArguments(
        output_dir=checkpoints_dir,
        evaluation_strategy="steps",
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=val_batch_size,
        predict_with_generate=True,
        logging_steps=500,  # set to 1000 for full training
        save_steps=save_steps,  # set to 500 for full training
        eval_steps=1000,  # set to 8000 for full training
        #warmup_steps=0,  # set to 2000 for full training
        #max_steps=40, # delete for full training
        #overwrite_output_dir=True,
        save_total_limit=2,
        fp16=False, 
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_df,
        eval_dataset=val_df,
    )
    trainer.train()
