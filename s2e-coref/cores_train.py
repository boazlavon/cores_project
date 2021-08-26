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

from transformers import AutoConfig, AutoTokenizer, CONFIG_MAPPING, LongformerConfig, BertConfig, BertTokenizer
from transformers import BertGenerationConfig, BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, EncoderDecoderConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainingArguments, Trainer
from transformers import T5ForConditionalGeneration

os.environ["PYTHONUNBUFFERED"] = '1'

batch_size=16
proj_dir = r'.'
data_dir = os.path.join(proj_dir, 'coref_data')

model_type = sys.argv[1]
if model_type not in ('bart', 'bert', 't5', 'init_t5', 'init_bert', 'init_bart'):
    print('Invalid Model Type')
    sys.exit(0)

model_type_no_init = model_type.replace('init_', '')
training_dataset_path = os.path.join(data_dir, f'{model_type_no_init}_train_dataset.pkl')
val_dataset_path = os.path.join(data_dir, f'{model_type_no_init}_val_dataset.pkl')
batch_size = int(sys.argv[2])
checkpoints_dir = os.path.join(proj_dir, f'{model_type}_{batch_size}_checkpoints')

init_w = False
if 'init' in model_type:
    init_w = True

latest_checkpoint = None
if os.path.isdir(checkpoints_dir):
    checkpoints = os.listdir(checkpoints_dir)
    if len(checkpoints):
        checkpoints.sort()
        latest_checkpoint = checkpoints[-1]
        latest_checkpoint = os.path.join(checkpoints_dir, latest_checkpoint)
latest_checkpoint = None

print(f'Training {model_type} Model!')
print(f'Loading Training Dataset: {training_dataset_path}')
with open(training_dataset_path, 'rb') as f:
    train_df = pickle.load(f)
print(f"Training: {len(train_df)}")

print(f'Loading Validation Dataset: {val_dataset_path}')
with open(val_dataset_path, 'rb') as f:
    val_df = pickle.load(f)
print(f"Validation: {len(val_df)}")

if 'bert' in model_type:
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
if 't5' in model_type:
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
if 'bart' in model_type:
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token
cores_tokens = ['<<', '>>', '[[u]]', '[[t]]', '[[f]]']
tokenizer.add_tokens(cores_tokens)
tokenizer.model_max_length = 128

if latest_checkpoint:
    print(f'Loading latest checkpoint: {latest_checkpoint}')
    if 'bert' in model_type:
        model = EncoderDecoderModel.from_pretrained(latest_checkpoint)
    elif 't5' in model_type:
        model = T5ForConditionalGeneration.from_pretrained(latest_checkpoint)
    elif 'bart' in model_type:
        model = BartForConditionalGeneration.from_pretrained(latest_checkpoint)
else:
    print(f'Building new {model_type} from pre-trained model')
    if 'bert' in model_type:
        encoder = BertGenerationEncoder.from_pretrained("bert-base-uncased")
        decoder = BertGenerationDecoder.from_pretrained("bert-base-uncased", add_cross_attention=True, is_decoder=True)
        encoder.resize_token_embeddings(len(tokenizer))
        decoder.resize_token_embeddings(len(tokenizer))
        if init_w:
            print('Init Pre-Trained Model Weights')
            encoder.init_weights()
            decoder.init_weights()
        model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
        model.config.vocab_size = model.config.decoder.vocab_size
        model.config.decoder_start_token_id = tokenizer.bos_token_id
    elif 't5' in model_type:
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        model.resize_token_embeddings(len(tokenizer))
        if init_w:
            print('Init Pre-Trained Model Weights')
            model.init_weights()
    elif 'bart' in model_type:
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
        model.resize_token_embeddings(len(tokenizer))
        if init_w:
            print('Init Pre-Trained Model Weights')
            model.init_weights()

from sklearn.metrics import precision_recall_fscore_support
def compute_metrics_f1(pred):
    labels = pred.label_ids
    preds = pred.predictions

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

if 'bart' in model_type:
    logging_dir=os.path.join(checkpoints_dir, 'logs')
    try:
        os.mkdir(logging_dir)
    except:
        pass
    batch_size = int(sys.argv[2])
    training_args = TrainingArguments(
                        output_dir=checkpoints_dir,          
                        evaluation_strategy="steps",
                        num_train_epochs=1,           
                        per_device_train_batch_size=batch_size, 
                        per_device_eval_batch_size=1,   
                        logging_steps=500,  # set to 1000 for full training
                        save_steps=2500,  # set to 500 for full training
                        warmup_steps=500,               
                        weight_decay=0.01,              
                        logging_dir=logging_dir,
                        eval_steps=50000,  # set to 8000 for full training
                        save_total_limit=5,
                        overwrite_output_dir=True
                        )
    trainer = Trainer(
                  model=model,                       
                  args=training_args,                  
                  train_dataset=train_df,
                  eval_dataset=val_df,
                  compute_metrics=compute_metrics_f1
    )
    trainer.train()
    sys.exit(0)

# Generic configs
# set special tokens
# model.config.eos_token_id = tokenizer.eos_token_id
# model.config.pad_token_id = tokenizer.pad_token_id

# sensible parameters for beam search
#model.config.max_length = 128
#model.config.early_stopping = True
#model.config.num_beams = 4
#model.config.min_length = 40
#model.config.no_repeat_ngram_size = 3
#model.config.length_penalty = 2.0

train_batch_size=batch_size
val_batch_size=batch_size
save_steps=500
if 'bart' in model_type:
    train_batch_size=2
    val_batch_size=2
    save_steps=5000

print(f'train_batch_size={train_batch_size}, val_batch_size={val_batch_size}')
# set training arguments - these params are not really tuned, feel free to change
training_args = Seq2SeqTrainingArguments(
    output_dir=checkpoints_dir,
    evaluation_strategy="steps",
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=val_batch_size,
    predict_with_generate=True,
    logging_steps=1000,  # set to 1000 for full training
    save_steps=save_steps,  # set to 500 for full training
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
    compute_metrics=compute_metrics_f1,
    train_dataset=train_df,
    eval_dataset=val_df,
)
trainer.train()
