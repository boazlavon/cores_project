from transformers import BertTokenizerFast
from transformers import T5Tokenizer
from cores_tokens import CoresDatasetPreProcessor
import datasets
from datasets import Dataset, concatenate_datasets
import pickle
import random
import os
import sys

from transformers import AutoConfig, AutoTokenizer, CONFIG_MAPPING, LongformerConfig, BertConfig, BertTokenizer
from transformers import BertGenerationConfig, BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, EncoderDecoderConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration

os.environ["PYTHONUNBUFFERED"] = '1'

batch_size=16
proj_dir = r'.'
data_dir = os.path.join(proj_dir, 'bert2bert_coref_data')

model_type = sys.argv[1]
if model_type not in ('bert', 't5'):
    print('Invalid Model Type')
    sys.exit(0)

training_dataset_path = os.path.join(data_dir, f'{model_type}_train_dataset.pkl')
val_dataset_path = os.path.join(data_dir, f'{model_type}_val_dataset.pkl')
checkpoints_dir = os.path.join(proj_dir, f'{model_type}_checkpoints')
cache_dir = os.path.join(proj_dir, 'bert2bert_cache')

latest_checkpoint = None
if os.path.isdir(checkpoints_dir):
    checkpoints = os.listdir(checkpoints_dir)
    if len(checkpoints):
        checkpoints.sort()
        latest_checkpoint = checkpoints[-1]
        latest_checkpoint = os.path.join(checkpoints_dir, latest_checkpoint)

print(f'Training {model_type} Model!')
print(f'Loading Training Dataset: {training_dataset_path}')
with open(training_dataset_path, 'rb') as f:
    train_df = pickle.load(f)
print(f"Training: {len(train_df)}")

print(f'Loading Validation Dataset: {val_dataset_path}')
with open(val_dataset_path, 'rb') as f:
    val_df = pickle.load(f)
print(f"Validation: {len(val_df)}")

if model_type == 'bert':
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", cache_dir=cache_dir)
if model_type == 't5':
    tokenizer = T5Tokenizer.from_pretrained("t5-small", cache_dir=cache_dir)

tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token
cores_tokens = ['<<', '>>', '[[u]]', '[[t]]', '[[f]]']
tokenizer.add_tokens(cores_tokens)
tokenizer.model_max_length = 128

if latest_checkpoint:
    print(f'Loading latest checkpoint: {latest_checkpoint}')
    model = EncoderDecoderModel.from_pretrained(latest_checkpoint)
else:
    print(f'Building new {model_type} model')
    if model_type == 'bert':
        encoder = BertGenerationEncoder.from_pretrained("bert-base-uncased", cache_dir=cache_dir)
        decoder = BertGenerationDecoder.from_pretrained("bert-base-uncased", cache_dir=cache_dir, add_cross_attention=True, is_decoder=True)
        encoder.resize_token_embeddings(len(tokenizer))
        decoder.resize_token_embeddings(len(tokenizer))
        model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
        model.config.vocab_size = model.config.decoder.vocab_size
    elif model_type == 't5':
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        model.resize_token_embeddings(len(tokenizer))

    # Generic configs
    # set special tokens
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # sensible parameters for beam search
    model.config.max_length = 128
    model.config.min_length = 40
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

# load rouge for validation
rouge = datasets.load_metric("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids < 0] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

# set training arguments - these params are not really tuned, feel free to change
training_args = Seq2SeqTrainingArguments(
    output_dir=checkpoints_dir,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    logging_steps=1000,  # set to 1000 for full training
    save_steps=500,  # set to 500 for full training
    eval_steps=8000,  # set to 8000 for full training
    warmup_steps=2000,  # set to 2000 for full training
    #max_steps=16, # delete for full training
    overwrite_output_dir=True,
    save_total_limit=3,
    fp16=False, 
)

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_df,
    eval_dataset=val_df,
)
trainer.train()
