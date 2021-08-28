from transformers import BertTokenizerFast
from transformers import T5Tokenizer
from cores_tokens import CoresDatasetPreProcessor
import datasets
from datasets import Dataset, concatenate_datasets
import pickle
import random
import os
import sys
import torch
import argparse
from cores_inference import inference

from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AutoConfig, AutoTokenizer, CONFIG_MAPPING, LongformerConfig, BertConfig, BertTokenizer
from transformers import BertGenerationConfig, BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, EncoderDecoderConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration

os.environ["PYTHONUNBUFFERED"] = '1'

def main():
    batch_size=16
    proj_dir = r'.'
    data_dir = os.path.join(proj_dir, 'coref_data')
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset_builder_path', type=str)
    parser.add_argument('--dropout', type=float)
    args = parser.parse_args(sys.argv[1:])
    model_type = args.model
    if model_type not in ('bert', 't5', 'bart', 'init_bart', 'init_t5'):
        print('Invalid Model Type')
        sys.exit(0)

    print("Loading tokenizer")
    if model_type == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token
    if 't5' in model_type:
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
    if 'bart' in model_type:
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token

    cores_tokens = ['<<', '>>', '[[u]]', '[[t]]', '[[f]]']
    tokenizer.add_tokens(cores_tokens)
    tokenizer.model_max_length = 128

    dataset_builder_path = args.dataset_builder_path
    print(f'Builder path: {dataset_builder_path}')
    if not os.path.exists(dataset_builder_path):
        print(f'Please generate {dataset_builder_path}')
        sys.exit(0)

    with open(dataset_builder_path, 'rb') as f:
        builder = pickle.load(f)

    config = f'{args.dropout}'
    checkpoints_dir = os.path.join(proj_dir, 'training_results', model_type, config)
    latest_checkpoint = None
    if os.path.isdir(checkpoints_dir) or os.path.islink(checkpoints_dir):
        checkpoints = os.listdir(checkpoints_dir)
        checkpoints = [ name for name in checkpoints if 'checkpoint' in name ]
        if len(checkpoints):
            checkpoints.sort()
            latest_checkpoint = checkpoints[-1]
            latest_checkpoint = os.path.join(checkpoints_dir, latest_checkpoint)
    if not latest_checkpoint:
        print('Please train {model_type}')
        sys.exit(0)

    print(f'Latest checkpoint: {latest_checkpoint}')
    if model_type == 'bert':
        print(f'Loading latest checkpoint: {latest_checkpoint}')
        model = EncoderDecoderModel.from_pretrained(latest_checkpoint)
    if 't5' in model_type:
        model = T5ForConditionalGeneration.from_pretrained(latest_checkpoint)
    elif 'bart' in model_type:
        model = BartForConditionalGeneration.from_pretrained(latest_checkpoint)

    model.eval()
    beam_size=4
    x = True
    if x:
        inference_result = []
        invalid_examples = []
        for i in range(3):
            print(f'cluster {i}')
            input_str = builder.clusters_df['input_str'][i]
            y = builder.clusters_df['output_str'][i]
            print('Input')
            print(input_str)
            print('Target')
            print(y)
            print()
            model.config.min_length = len(tokenizer.encode(y))
            model.config.max_length = 128
            print(f'min length: {model.config.min_length}')
            input_str = [input_str,]
            inputs  = tokenizer(input_str,  padding="max_length", truncation=True, max_length=128)
            outputs = model.generate(torch.tensor(inputs.input_ids), attention_mask=torch.tensor(inputs.attention_mask),
                                     num_beams=5, num_return_sequences=1, no_repeat_ngram_size=2)
            output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            #output_str = output_str[0]
            print('Model Output')
            print(output_str)
            print()
            continue
            try:
                mentions_envs, mentions_only = inference(output_str)
            except:
                mentions_envs, mentions_only = [], []
            print(f'Envs: {len(mentions_envs)}\nOnly: {len(mentions_only)}')
            inference_result.append({'tag'           : f'cluster_{i}', 
                                     'input_str'     : input_str, 
                                     'output_str'    : output_str, 
                                     'mentions_envs' : mentions_envs, 
                                     'mentions_only' : mentions_only})
            if (len(mentions_envs) != len(mentions_only)):
                invalid_examples.append({'tag'          : f'cluster_{i}', 
                                         'input_str'     : input_str, 
                                         'output_str'    : output_str, 
                                         'mentions_envs' : mentions_envs, 
                                         'mentions_only' : mentions_only})
            print()

        for i in range(3):
            print(f'mention: {i}')
            input_str = builder.mentions_df['input_str'][i]
            y = builder.mentions_df['output_str'][i]
            print('Input')
            print(input_str)
            print('Target')
            print(y)
            print()
            model.config.min_length = len(tokenizer.encode(y))
            model.config.max_length = 128
            print(f'min length: {model.config.min_length}')
            input_str = [input_str,]
            inputs  = tokenizer(input_str,  padding="max_length", truncation=True, max_length=128)
            outputs = model.generate(torch.tensor(inputs.input_ids), attention_mask=torch.tensor(inputs.attention_mask),
                                     num_beams=beam_size ,num_return_sequences=1, no_repeat_ngram_size=0)
                                     #return_dict_in_generate=True, output_scores=True)
            output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_str = output_str[0]
            print('Model Output')
            print(output_str)
            print()
            print()
            continue
            try:
                mentions_envs, mentions_only = inference(output_str)
            except:
                mentions_envs, mentions_only = [], []
            inference_result.append({'tag'          : f'mention_{i}', 
                                     'input_str'     : input_str, 
                                     'output_str'    : output_str, 
                                     'mentions_envs' : mentions_envs, 
                                     'mentions_only' : mentions_only})
            if (len(mentions_envs) != len(mentions_only)):
                print('Invalid!')
                invalid_examples.append({'tag'          : f'mention_{i}', 
                                         'input_str'     : input_str, 
                                         'output_str'    : output_str, 
                                         'mentions_envs' : mentions_envs, 
                                         'mentions_only' : mentions_only})

        print(f'Exit!')
        print(f'Inference Results: {len(inference_result)}')
        print(f'Invalid Examples: {len(invalid_examples)}')
        for item in invalid_examples:
            print(item['tag'])
            print(item['input_str'])
            print(f'Mentions with envs: {len(item["mentions_envs"])}') 
            print(f'Mentions only: {len(item["mentions_only"])}')
            print()

if __name__ == '__main__':
    main()
