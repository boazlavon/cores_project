from cores_tokens import CoresDatasetPreProcessor
from transformers import BertTokenizerFast
from transformers import T5Tokenizer, BartTokenizer
from datasets import Dataset, concatenate_datasets
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers import BartForConditionalGeneration, BartTokenizer

import datasets
import pickle
import random
import os
import sys

test_size = 0.1
seed = 42
random.seed(seed)
batch_size=16  # change to 16 for full training
encoder_max_length=128
decoder_max_length=128

proj_dir = r'.'
data_dir = os.path.join(proj_dir, 'coref_data')

model_type = sys.argv[1]
if model_type not in ('bert', 'bart'):
    print('Invalid Model Type')
    sys.exit(0)

training_builder_path = sys.argv[2]
if not os.path.exists(training_builder_path):
    print(f'Training Builder: {training_builder_path} dont exist')
    sys.exit(0)

val_builder_path = sys.argv[3]
if not os.path.exists(val_builder_path):
    print(f'Validation Builder: {val_builder_path} dont exist')
    sys.exit(0)

training_dataset_path = os.path.join(data_dir, f'{model_type}_train_dataset.pkl')
val_dataset_path = os.path.join(data_dir, f'{model_type}_val_dataset.pkl')
if os.path.exists(training_dataset_path):
    print(f'{training_dataset_path} already exists')
    sys.exit(0)

if os.path.exists(val_dataset_path):
    print(f'{val_dataset_path} already exists')
    sys.exit(0)
if model_type == 'bert':
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token
if model_type == 't5':
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
if model_type == 'bart':
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

cores_tokens = ['<<', '>>', '[[u]]', '[[t]]', '[[f]]']
tokenizer.add_tokens(cores_tokens)
tokenizer.model_max_length = 128

try:
    print("Loading Training Pre-Processor")
    with open(training_builder_path, 'rb') as f:
        train_builder = pickle.load(f)
except:
    print(f"Please building New Pre-Processor: {training_builder_path} cores_tokens.py/cores_tokens_test.py")

try:
    print("Loading Validation Pre-Processor")
    with open(val_builder_path, 'rb') as f:
        val_builder = pickle.load(f)
except:
    print(f"Please building New Pre-Processor: {training_builder_path} cores_tokens.py/cores_tokens_test.py")

print(f"Building Training & Validation Datasets for {model_type}")
print("Split Training & Validation")

# Make sure that same chunks are used in mentions and clusters validation & training.
mentions_df_train  = Dataset.from_pandas(train_builder.mentions_df)
clusters_df_train  = Dataset.from_pandas(train_builder.clusters_df)
mentions_df_val  = Dataset.from_pandas(val_builder.mentions_df)
clusters_df_val  = Dataset.from_pandas(val_builder.clusters_df)

if model_type == 'bart':
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base', cache_dir='./cache')

def bart_process_data(example_batch):
    global model
    input_encodings  = tokenizer.batch_encode_plus(example_batch['input_str'],  padding="max_length", truncation=True, 
                                                   max_length=encoder_max_length, return_tensors="pt")
    target_encodings = tokenizer.batch_encode_plus(example_batch['output_str'], padding="max_length", truncation=True, 
                                                   max_length=decoder_max_length, return_tensors="pt")
                
    labels = target_encodings['input_ids']
    decoder_input_ids = shift_tokens_right(labels, model.config.pad_token_id, model.config.decoder_start_token_id)
    labels[labels[:, :] == model.config.pad_token_id] = -100
                                
    encodings = {
        'input_ids': input_encodings['input_ids'].tolist(),
        'attention_mask': input_encodings['attention_mask'].tolist(),
        'decoder_input_ids': decoder_input_ids.tolist(),
        'labels': labels.tolist(),
    }
    return encodings

def bert_process_data(batch):
    # tokenize the inputs and labels
    inputs  = tokenizer(batch["input_str"],  padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch["output_str"], padding="max_length", truncation=True, max_length=decoder_max_length)
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()
    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]
    return batch

conver_func=bert_process_data
final_columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"]
if model_type == 'bart':
    conver_func=bart_process_data
    final_columns.remove("decoder_attention_mask")

print("Converting to tensors")
mentions_df_train = mentions_df_train.map(conver_func, batched=True, remove_columns=['idx', 'input_str', 'output_str'])
mentions_df_train.set_format(type="torch", columns=final_columns)

mentions_df_val = mentions_df_val.map(conver_func, batched=True, remove_columns=['idx', 'input_str', 'output_str'])
mentions_df_val.set_format(type="torch", columns=final_columns)

clusters_df_train = clusters_df_train.map(conver_func, batched=True, remove_columns=['idx', 'cluster_index', 'mention', 'input_str', 'output_str'])
clusters_df_train.set_format(type="torch", columns=final_columns)

clusters_df_val = clusters_df_val.map(conver_func, batched=True, remove_columns=['idx', 'cluster_index', 'mention', 'input_str', 'output_str'])
clusters_df_val.set_format(type="torch", columns=final_columns)

print(mentions_df_train["input_ids"].shape)
print(mentions_df_val["input_ids"].shape)
print(clusters_df_train["input_ids"].shape)
print(clusters_df_val["input_ids"].shape)
print()

train_df = datasets.concatenate_datasets([mentions_df_train, clusters_df_train])
val_df = datasets.concatenate_datasets([mentions_df_val, clusters_df_val])
print(train_df["input_ids"].shape)
print(val_df["input_ids"].shape)

print(f"Save Final Training & Validation Datasets for {model_type}")
training_dataset_path = os.path.join(data_dir, f'{model_type}_train_dataset.pkl')
val_dataset_path = os.path.join(data_dir, f'{model_type}_val_dataset.pkl')
with open(training_dataset_path, 'wb') as f:
    pickle.dump(train_df, f)

with open(val_dataset_path, 'wb') as f:
    pickle.dump(val_df, f)
