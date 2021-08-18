#ls -la ./bert2bert_coref_data/train.english.jsonlines
#python cores_tokens.py

from transformers import BertTokenizerFast
from cores_tokens import CoresDatasetPreProcessor
from transformers import T5Tokenizer
from datasets import Dataset, concatenate_datasets
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
data_dir = os.path.join(proj_dir, 'bert2bert_coref_data')
cache_dir = os.path.join(proj_dir, 'bert2bert_cache')

model_type = sys.argv[1]
if model_type not in ('bert', 't5'):
    print('Invalid Model Type')
    sys.exit(0)

training_file = 'train.english.jsonlines'
training_data_path = os.path.join(data_dir, training_file)
dataset_builder_path = f'{training_data_path}.builder.{model_type}.pkl'

training_dataset_path = os.path.join(data_dir, f'{model_type}_{training_file}_train_dataset.pkl')
val_dataset_path = os.path.join(data_dir, f'{model_type}_{training_file}_val_dataset.pkl')
if os.path.exists(training_dataset_path):
    print(f'{training_data_path} already exists')
    sys.exit(0)

if model_type == 'bert':
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", cache_dir=cache_dir)
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token
if model_type == 't5':
    tokenizer = T5Tokenizer.from_pretrained("t5-small", cache_dir=cache_dir)

cores_tokens = ['<<', '>>', '[[u]]', '[[t]]', '[[f]]']
tokenizer.add_tokens(cores_tokens)
tokenizer.model_max_length = 128

if os.path.exists(dataset_builder_path):
    with open(dataset_builder_path, 'rb') as f:
        print("Loading Pre-Processor")
        builder = pickle.load(f)
else:
    print(f"Building New Pre-Processor: {dataset_builder_path}")
    builder = CoresDatasetPreProcessor(training_data_path, tokenizer, max_seq_length=decoder_max_length)
    with open(dataset_builder_path, 'wb') as f:
        pickle.dump(builder, f)
        print(f"Saved: {dataset_builder_path}")

print(f"Building Training & Validation Datasets for {model_type}")
print("Split Training & Validation")
# Make sure that same chunks are used in mentions and clusters validation & training.
mentions_df  = Dataset.from_pandas(builder.mentions_df)
clusters_df  = Dataset.from_pandas(builder.clusters_df)
mentions_df  = mentions_df.select(range(30))
clusters_df  = clusters_df.select(range(30))
mentions_idxs = mentions_df['idx']
random.shuffle(mentions_idxs)
train_idxs_count = int((1 - test_size) * len(mentions_idxs))
train_idxs = mentions_idxs[:train_idxs_count]
val_idxs   = mentions_idxs[train_idxs_count:]

mentions_df_train = mentions_df.filter(lambda example : example['idx'] in train_idxs )
print(len(mentions_df_train))
mentions_df_val   = mentions_df.filter(lambda example : example['idx'] in val_idxs )
print(len(mentions_df_val))

clusters_df_train = clusters_df.filter(lambda example : example['idx'] in train_idxs )
print(len(clusters_df_train))

clusters_df_val   = clusters_df.filter(lambda example : example['idx'] in val_idxs )
print(len(clusters_df_val))

# Validate the clusters df
print((1.0 * len(clusters_df_val)) / (len(clusters_df_train) + len(clusters_df_val)))
if (set(clusters_df_val["idx"]) & set(clusters_df_train["idx"])):
    raise ValueError('Invalid Split!')

def process_cores_data_to_model_inputs(batch):
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

print("Converting to tensors")
mentions_df_train = mentions_df_train.map(
    process_cores_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=['idx', 'input_str', 'output_str']
)
mentions_df_train.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

mentions_df_val = mentions_df_val.map(
    process_cores_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=['idx', 'input_str', 'output_str']
)
mentions_df_val.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

clusters_df_train = clusters_df_train.map(
    process_cores_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=['idx', 'cluster_index', 'mention', 'input_str', 'output_str']
)
clusters_df_train.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

clusters_df_val = clusters_df_val.map(
    process_cores_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=['idx', 'cluster_index', 'mention', 'input_str', 'output_str']
)
clusters_df_val.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

print(clusters_df_train)
print(clusters_df_val)

print(mentions_df_train["input_ids"].shape)
print(mentions_df_val["input_ids"].shape)
print(clusters_df_train["input_ids"].shape)
print(clusters_df_val["input_ids"].shape)
print()

print("Exdending Mentions Dataset")
# I want to make sure that durring training each batch will have equal amount of mentions and clusters examples.
print(len(mentions_df_train))
print(len(clusters_df_train))
rate = len(clusters_df_train) / len(mentions_df_train)
print(f"Clusters is bigger {rate} times that mentions")
rate = int(rate / 2.0)
print(f"Extending mentions by {rate}")

extended_mentions_df = mentions_df_train
for i in range(rate):
    extended_mentions_df = concatenate_datasets([extended_mentions_df, mentions_df_train])

print(len(extended_mentions_df))
print(len(clusters_df_train))
print(int(len(clusters_df_train) / len(extended_mentions_df)))

train_df = datasets.concatenate_datasets([extended_mentions_df, clusters_df_train])
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
