import json
import random
import logging
import os
import pickle
import time, threading, sys
import re
import difflib

import datasets
from datasets import Dataset, concatenate_datasets
from datasets import Dataset, load_metric

import torch
import pandas as pd
from cores_tokens import UNK_CLUSTER_TOKEN, IN_CLUSTER_TOKEN, NOT_IN_CLUSTER_TOKEN

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
CAPTURE_MENTIONS_ENVS_RE = '(?P<lenv>([^\s]+ ){0,3})<< (?P<span>([^\s<>]+ )*[^\s]+) >> \[\[(?P<cluster_tag>[uft])\]\][^\s]?(?P<renv>( [^\s<]+){0,3})'
CAPTURE_ONLY_MENTION_RE = '<< (?P<span>([^\s<>]+ )*[^\s]+) >> \[\[(?P<cluster_tag>[uft])\]\]'
#CAPTURE_MENTIONS_ENVS_RE = '(?P<lenv>([^\\s]+ ){0,3})<< (?P<span>(\\w+\\s)*\\w+) >> \\[\\[(?P<cluster_tag>[uft])\\]\\][\?]?(?P<renv>( [^\\s<]+){0,3})'
#CAPTURE_ONLY_MENTION_RE = '<< (?P<span>(\\w+\\s)*\\w+) >> \\[\\[(?P<cluster_tag>[uft])\\]\\]'

def extract_mentions(sentence):
    mentions = []
    m_iter = re.finditer(CAPTURE_ONLY_MENTION_RE, sentence)
    count = 1
    for m in m_iter:
        print(f'mention: {count}')
        mention = { 'span'        : m['span'], 
                    'cluster_tag' : m['cluster_tag'] }
        print(f'("{mention["span"]}" || {mention["cluster_tag"]})')
        print()
        mentions.append(mention)
        count += 1
    count -= 1
    print(f'total mentions: {count}')
    return mentions

def clean_span(span):
    span = span.strip()
    EXCLUDE_TOKENS = '><'
    for token in EXCLUDE_TOKENS:
        span = span.replace(token, '')
    span = span.strip()

    INCLUDE_TOKENS = ','
    for token in INCLUDE_TOKENS:
        span = span.replace(token, ' ' + token)
    return span

MEN_LENV_IDX = 0
MEN_RENV_IDX = 1
MEN_SPAN_STR_IDX = 2
MEN_SPAN_RANGE_IDX = 3
MEN_CLUSTER_TAG_IDX = 4

def extract_mentions_with_env(sentence):
    mentions = []
    m_iter = re.finditer(CAPTURE_MENTIONS_ENVS_RE, sentence)
    count = 1
    for m in m_iter:
        textual_span = clean_span(m['span']).strip()
        mention = (m['lenv'].strip(), m['renv'].strip(), textual_span, m.span(), m['cluster_tag'])
        mentions.append(mention)
        count += 1
    count -= 1
    return mentions

M19 = "so, how exactly did citizens react? let's take a look at the reactions of citizens. - - an sms. << i >> [[u]] saw << it >> [[u]] was that jingguang bridge. hey, i say, this is quite good, quite good. have << you >> [[u]] received information like << this >> [[u]] before? no, no. this was the first time. << the first times? >> [[u]] yes, yes. well, what do you think of the speed at which << the government >> [[u]] responded << this time >> [[u]]? quite fast. really quite fast"

M18 = 'using the most familiar method, most used by everyone, as well as the most prompt and most convenient method, for the announcement. yes, << i >> [[u]] noticed that many friends, around << me >> [[u]] received << it >> [[u]]. it seems that almost everyone received << this sms >> [[u]] from yes. yes. the effect was extremely good, ha.'
def inference(sentence):
    print('MENTIONS WITH ENVS')
    mentions_envs = extract_mentions_with_env(sentence)
    print('MENTIONS ONLY')
    mentions_only = extract_mentions(sentence)
    return (mentions_envs, mentions_only)

def load_pickles():
    os.environ["PYTHONUNBUFFERED"] = '1'

    proj_dir = r'.'
    data_dir = os.path.join(proj_dir, 'bert2bert_coref_data')

    beam_size = sys.argv[3]
    beam_size = int(beam_size)
    if not ((beam_size > 0) and (beam_size < 5)):
        print('Beam-Size out of range')
        sys.exit(0)

    model_type = sys.argv[1]
    if model_type not in ('bert', 't5', 'init_bert', 'init_t5'):
        print('Invalid Model Type')
        sys.exit(0)

    print("Loading tokenizer")
    if model_type == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token
    if model_type == 't5':
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
    cores_tokens = ['<<', '>>', '[[u]]', '[[t]]', '[[f]]']
    tokenizer.add_tokens(cores_tokens)
    tokenizer.model_max_length = 128

    raw_data_path = sys.argv[2]
    if not os.path.exists(raw_data_path):
        print(f'Raw Data file isn\'t exist: {raw_data_path}')
        sys.exit(0)

    raw_data_file = os.path.basename(raw_data_path)
    is_test = 'test' in raw_data_file
    
    print(f'Raw data file: {raw_data_file} is_test={is_test}')
    dataset_builder_path = f'{raw_data_path}.builder.{model_type}.pkl'
    print(f'Builder path: {dataset_builder_path}')
    if not os.path.exists(dataset_builder_path):
        print(f'Please generate builder (data_preprocessor.py script): {dataset_builder_path}')
        sys.exit(0)

    print("Loading Builder")
    with open(dataset_builder_path, 'rb') as f:
        builder = pickle.load(f)

    checkpoints_dir = os.path.join(proj_dir, f'{model_type}_checkpoints')
    latest_checkpoint = None
    if os.path.isdir(checkpoints_dir):
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
    if model_type == 't5':
        model = T5ForConditionalGeneration.from_pretrained(latest_checkpoint)

    return builder, tokenizer, model

def generate_true_cluster_example(true_mention, sentence, model_output_mentions):
    for m in model_output_mentions:
        replace_tok = UNK_CLUSTER_TOKEN
        if m[MEN_SPAN_RANGE_IDX] == true_mention[MEN_SPAN_RANGE_IDX]:
            replace_tok = IN_CLUSTER_TOKEN
        start, end = m[MEN_SPAN_RANGE_IDX]
        full_mention = sentence[start: end + 1]
        if f'[[{m[MEN_CLUSTER_TAG_IDX]}]]' == replace_tok:
            continue
        replaced_full_mention = full_mention.replace(f'[[{m[MEN_CLUSTER_TAG_IDX]}]]', replace_tok)
        if (full_mention != replaced_full_mention):
            print(f'Replace!: {m}')
            print(full_mention)
            print(replaced_full_mention)
            sentence = sentence.replace(full_mention, replaced_full_mention)
        return sentence

def create_suffix_map(words, words_str):
    words = [(idx, word) for idx, word in enumerate(words)]
    suffix_map = {}
    substrings = []
    for idx, word in words[::-1]:
        suffix = [word for _, word in words[idx:]]
        suffix = ' '.join(suffix)
        suffix_idx = words_str.find(suffix, len(words_str) - len(suffix), len(words_str))
        suffix_map[suffix_idx] = idx
    return suffix_map

def execute_model(input_str, model, tokenizer, beam_size):
    input_str = [input_str,]
    inputs  = tokenizer(input_str,  padding="max_length", truncation=True, max_length=128)
    model_outputs = model.generate(torch.tensor(inputs.input_ids), attention_mask=torch.tensor(inputs.attention_mask), num_beams=beam_size ,num_return_sequences=1, no_repeat_ngram_size=0) #return_dict_in_generate=True, output_scores=True)
    model_output_str = tokenizer.batch_decode(model_outputs, skip_special_tokens=True)
    model_output_str = model_output_str[0]
    return model_output_str

def inference_example(model, builder, tokenizer, i, idx, chunk_id, words, clusters, mentions_env, span_mention_lookup, model_output_str=None):

    beam_size = int(sys.argv[3])
    model.config.no_repeat_ngram_size = None
    # Suprise ! put the output mentions and check of good it good clustering only
    if model_output_str is None:
        # execute the model
        input_str = ' '.join(words)
        input_str = input_str.lower()
        print('Input String')
        print(input_str)
        model_output_str = execute_model(input_str, model, tokenizer, beam_size)
        print()
        print('Model Output')
        print(model_output_str)

    # extract mentions from model output
    model_mentions_string = model_output_str
    model_output_mentions = extract_mentions_with_env(model_mentions_string)
    print(model_output_mentions)

    # update mentions 
    pred_clusters = {}

    cluster_pred_outputs = {}
    # for each mention
    for j, mention in enumerate(model_output_mentions):
        print(f'Mention ({j}): {mention}')

        # replace it to a cluster example
        true_cluster_sentence = generate_true_cluster_example(mention, model_mentions_string, model_output_mentions)
        print(f'Generate Cluster Sentence')
        print(true_cluster_sentence)

        # execute the model and get the cluster tagging
        model_output_str = execute_model(true_cluster_sentence, model, tokenizer, beam_size)
        print('Cluster Tagging from Model:')
        print(model_output_str)
        print()
        cluster_pred_outputs[mention] = model_output_str

    # for each mention
    for j, mention in enumerate(model_output_mentions):
        print(f'Mention ({j}): {mention}')

        model_output_str = cluster_pred_outputs[mention]

        # extract mentions and taggings from output
        model_output_mentions = extract_mentions_with_env(model_output_str)

        # update_clusters
        pred_clusters[mention] = [ m for m in model_output_mentions if m[MEN_CLUSTER_TAG_IDX] == 't' ]
        print(pred_clusters[mention])
    return pred_clusters, cluster_pred_outputs

import pickle
def save_clusters(results, raw_data_path, idx=0):
    proj_dir = r'.'
    data_dir = os.path.join(proj_dir, 'bert2bert_coref_data')
    path = os.path.join(data_dir, f'inference_results.{raw_data_path}.{idx}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(results, f)

def load_clusters(raw_data_path, idx=0):
    with open(path, 'rb') as f:
        return pickle.load(f)

def choose_by_env(mention, span_idxs, suffix_map, words, env_size=3):
    mention[MEN_LENV_IDX]
    mention[MEN_RENV_IDX]

    words_starts = {}
    chosen = None
    max_score = -1
    print(f'Candidate: {mention}')
    for idx in span_idxs:
        if idx in suffix_map:
            start = suffix_map[idx]
            words_count = len(mention[MEN_SPAN_STR_IDX].split(' '))
            end = start + (words_count - 1)
            span = ' '.join(words[start : end + 1])
            renv = ' '.join(words[end + 1 : end + 1 + env_size])
            lenv = ' '.join(words[max(0, start - env_size) : start])
            print()
            print((start, end))
            print(span)
            print(renv)
            print(lenv)
            renv_score = difflib.SequenceMatcher(None, renv, mention[MEN_RENV_IDX]).ratio()
            lenv_score = difflib.SequenceMatcher(None, lenv, mention[MEN_LENV_IDX]).ratio()
            score = renv_score + lenv_score
            print(score)
            if score > max_score:
                chosen = (start, end)
                max_score   = score
    print('====')
    print(mention[MEN_LENV_IDX])
    print(mention[MEN_RENV_IDX])
    print(chosen)
    print('====')
    return chosen

def create_suffix_map(words, words_str):
    words = [(idx, word) for idx, word in enumerate(words)]
    suffix_map = {}
    substrings = []
    for idx, word in words[::-1]:
        suffix = [word for _, word in words[idx:]]
        suffix = ' '.join(suffix)
        suffix_idx = words_str.find(suffix, len(words_str) - len(suffix), len(words_str))
        suffix_map[suffix_idx] = idx
    return suffix_map

def match_mention(mention, words, words_str, suffix_map):
    i = -1
    span_str = mention[MEN_SPAN_STR_IDX].lower()
    found_idxs = []
    while i < len(words_str):
        span_idx = words_str.find(span_str, i + 1, len(words_str))
        if span_idx == -1:
            break;
        found_idxs.append(span_idx)
        i = span_idx
    return found_idxs


def match_mention_to_word(mention , words):
    words_str = ' '.join(words).lower()
    suffix_map = create_suffix_map(words, words_str)
    span_idxs = match_mention(mention, words, words_str, suffix_map)
    if len(span_idxs) == 0:
        start = -1
    elif len(span_idxs) == 1:
        start = span_idxs[0]
        start = suffix_map[start]
        words_count = len(mention[MEN_SPAN_STR_IDX].split(' '))
        end = start + (words_count - 1)
    else: # len > 1
        result = choose_by_env(mention, span_idxs, suffix_map, words)
        if result is None:
            start, end = -1, -1
        else:
            start, end = result
    if start == -1 or end == -1:
        return None
    return (start, end)

def create_seperate_clusters(pred_clusters_golden_idxs):
    clusters = [set(list(items) + [key]) for key, items in pred_clusters_golden_idxs.items()]
    i = 0
    while i < len(clusters):
        j = i + 1
        while j < len(clusters):
            if clusters[i] & clusters[j]:
                clusters[i] = clusters[i].union(clusters[j])
                del clusters[j]
                j = i + 1
                continue
            j += 1
        i += 1
    clusters = [list(c) for c in clusters]
    return clusters

def predict_final_clusters(pred_clusters, words):
    pred_clusters_golden_idxs = {}
    for mention in pred_clusters.keys():
        match_result = match_mention_to_word(mention, words)
        if match_result is None:
            print(f'Could not find {mention}')
            continue
        start, end = match_result
        pred_clusters_golden_idxs[(start, end)] = []
        main_key = (start, end)

        for mention in pred_clusters[mention]:
            match_result = match_mention_to_word(mention, words)
            if match_result is None:
                print(f'Could not find {mention}')
                continue
            start, end = match_result
            pred_clusters_golden_idxs[main_key].append((start, end))

    final_clusters = create_seperate_clusters(pred_clusters_golden_idxs)
    return final_clusters

def generate_inference_results(builder, tokenizer, model):
    model_type = sys.argv[1]
    raw_data_path = sys.argv[2]
    beam_size = int(sys.argv[3])

    raw_data_path = os.path.basename(raw_data_path)
    results = []
    proj_dir = r'.'
    data_dir = os.path.join(proj_dir, 'bert2bert_coref_data')
    infer_dir = os.path.join(data_dir, f'inference_results.{model_type}.{raw_data_path}.b{beam_size}')
    try:
        os.mkdir(infer_dir)
    except:
        pass

    # for each example in test (builder)
    for i, (idx, chunk_id, words, golden_clusters, mentions_env, span_mention_lookup) in enumerate(builder.env_examples):
        results_path = os.path.join(infer_dir, f'{idx}_{chunk_id}.pkl')
        try:
            with open(results_path, 'rb') as f:
                pred_clusters, cluster_pred_outputs, final_clusters = pickle.load(f)
            print(f'Loaded {idx}: {results_path}')
            continue

        except:
            print(f'Infering {idx}_{chunk_id}')
            words = [w.lower() for w in words]
            pred_clusters, cluster_pred_outputs = inference_example(model, builder, tokenizer, i, idx, chunk_id, words, golden_clusters, mentions_env, span_mention_lookup)

            final_clusters = predict_final_clusters(pred_clusters, words)
            print('=======================')
            print('Predicted:')
            for cluster in final_clusters:
                values_str = tuple([' '.join(words[start : end + 1]) for start, end in cluster])
                print(f'{values_str}')
            print()
            print('Golden:')
            for cluster in golden_clusters:
                values_str = tuple([' '.join(words[start : end + 1]) for start, end in cluster])
                print(f'{values_str}')
            print('=======================')

            with open(results_path, 'wb') as f:
                pickle.dump((pred_clusters, cluster_pred_outputs, final_clusters), f)
                print(f'Saved Model! {results_path}')

def print_inference(builder, tokenizer, model):
    model_type = sys.argv[1]
    raw_data_path = sys.argv[2]
    raw_data_path = os.path.basename(raw_data_path)
    beam_size = int(sys.argv[3])

    results = []
    proj_dir = r'.'
    data_dir = os.path.join(proj_dir, 'bert2bert_coref_data')
    infer_dir = os.path.join(data_dir, f'inference_results.{model_type}.{raw_data_path}.b{beam_size}')

    for i, (idx, chunk_id, words, golden_clusters, mentions_env, span_mention_lookup) in enumerate(builder.env_examples):
        print(f'Infering {idx}_{chunk_id}')
        words = [w.lower() for w in words]
        try:
            results_path = os.path.join(infer_dir, f'{idx}_{chunk_id}.pkl')
            with open(results_path, 'rb') as f:
                pred_clusters, cluster_pred_outputs, final_clusters = pickle.load(f)
            print(f'Loaded {idx}_{chunk_id}: {results_path}')
        except:
            print(f'Error loding {idx}_{chunk_id}')
            continue

        print('=======================')
        print('Predicted:')
        for cluster in final_clusters:
            values_str = tuple([' '.join(words[start : end + 1]) for start, end in cluster])
            print(f'{values_str}')
        print()
        print('Golden:')
        for cluster in golden_clusters:
            values_str = tuple([' '.join(words[start : end + 1]) for start, end in cluster])
            print(f'{values_str}')
        print('=======================')
        print()

def main():
    builder, tokenizer, model = load_pickles()
    generate_inference_results(builder, tokenizer, model)
    print()
    print('Print Inference!')
    print_inference(builder, tokenizer, model)

if __name__ == '__main__':
    main()
