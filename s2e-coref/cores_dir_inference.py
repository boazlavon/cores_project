import json
import time
import hashlib
import base64
import random
import argparse
import logging
import os
import pickle
import time, threading, sys
import re
import difflib

import datasets
from datasets import Dataset, concatenate_datasets
from datasets import Dataset, load_metric

from utils import flatten_list_of_lists
import torch
import pandas as pd
from cores_tokens import UNK_CLUSTER_TOKEN, IN_CLUSTER_TOKEN, NOT_IN_CLUSTER_TOKEN
from cores_tokens_test import CoresDatasetPreProcessorTest

# Training imports
from transformers import T5ForConditionalGeneration, T5Tokenizer 
from transformers import BartForConditionalGeneration, BartTokenizer
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
CAPTURE_MENTIONS_ENVS_RE = '(?P<lenv>([^\s]+ ){0,3})<< (?P<span>([^\s<>]+ )*[^\s<>]+) >> \[\[(?P<cluster_tag>[uft])\]\][^\s]?(?P<renv>( [^\s<]+){0,3})'
CAPTURE_ONLY_MENTION_RE = '<< (?P<span>([^\s<>]+ )*[^\s]+) >> \[\[(?P<cluster_tag>[uft])\]\]'
#CAPTURE_MENTIONS_ENVS_RE = '(?P<lenv>([^\\s]+ ){0,3})<< (?P<span>(\\w+\\s)*\\w+) >> \\[\\[(?P<cluster_tag>[uft])\\]\\][\?]?(?P<renv>( [^\\s<]+){0,3})'
#CAPTURE_ONLY_MENTION_RE = '<< (?P<span>(\\w+\\s)*\\w+) >> \\[\\[(?P<cluster_tag>[uft])\\]\\]'

CUDA_DEVICE = torch.device('cuda')

def extract_mentions(sentence):
    # the extraction here should be kind of strict to the format.
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
    # support more mentions
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

def load_pickles(model_type, dataset_builder_path, beam_size, config):
    os.environ["PYTHONUNBUFFERED"] = '1'
    if not ((beam_size > 0) and (beam_size < 5)):
        print('Beam-Size out of range')
        sys.exit(0)

    if model_type not in ('bert', 't5', 'bart', 'init_bart', 'init_bert', 'init_t5'):
        print('Invalid Model Type')
        sys.exit(0)

    print(f'Builder path: {dataset_builder_path}')
    if not os.path.exists(dataset_builder_path):
        print(f'Please generate builder (cores_tokens_test.py script): {dataset_builder_path}')
        sys.exit(0)

    proj_dir = r'.'
    checkpoints_dir = os.path.join(proj_dir, f'training_results/{model_type}/{config}')
    latest_checkpoint = None
    if os.path.isdir(checkpoints_dir):
        checkpoints = os.listdir(checkpoints_dir)
        checkpoints = [ name for name in checkpoints if 'checkpoint' in name ]
        if len(checkpoints):
            checkpoints.sort()
            latest_checkpoint = checkpoints[-1]
            latest_checkpoint = os.path.join(checkpoints_dir, latest_checkpoint)
    if not latest_checkpoint:
        print('Please train {model_type} with {config}')
        sys.exit(0)

    print("Loading tokenizer")
    if 'bert' in model_type:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    if 't5' in model_type:
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
    if 'bart' in model_type:
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    if 't5' not in model_type:
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token
    cores_tokens = ['<<', '>>', '[[u]]', '[[t]]', '[[f]]']
    tokenizer.add_tokens(cores_tokens)
    tokenizer.model_max_length = 128

    print("Loading Builder")
    with open(dataset_builder_path, 'rb') as f:
        builder = pickle.load(f)

    print(f'Latest checkpoint: {latest_checkpoint}')
    if 'bert' in model_type:
        model = EncoderDecoderModel.from_pretrained(latest_checkpoint)
    elif 't5' in model_type:
        model = T5ForConditionalGeneration.from_pretrained(latest_checkpoint)
    elif 'bart' in model_type:
        model = BartForConditionalGeneration.from_pretrained(latest_checkpoint)
    model.to(CUDA_DEVICE)
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
        if word: # for empty words
            suffix = [word for _, word in words[idx:] if word]
            suffix = ' '.join(suffix).replace('  ', ' ')
            suffix_idx = words_str.find(suffix, len(words_str) - len(suffix), len(words_str))
            suffix_map[suffix_idx] = idx
    return suffix_map

def execute_model(input_str, model, tokenizer, beam_size):
    model.config.min_length = len(tokenizer.encode(input_str))
    model.config.max_length = 128
    input_str = [input_str,]
    inputs  = tokenizer(input_str,  padding="max_length", truncation=True, max_length=128)
    model_outputs = model.generate(torch.tensor(inputs.input_ids).to(CUDA_DEVICE), attention_mask=torch.tensor(inputs.attention_mask).to(CUDA_DEVICE), num_beams=beam_size ,num_return_sequences=1) #return_dict_in_generate=True, output_scores=True)
    model_output_str = tokenizer.batch_decode(model_outputs, skip_special_tokens=True)
    model_output_str = model_output_str[0]
    return model_output_str

def inference_example(model, tokenizer, words, beam_size, model_output_str=None):
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
    #print(model_output_mentions)

    # update mentions 
    pred_obj_clusters = {}

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
        pred_obj_clusters[mention] = [ m for m in model_output_mentions if m[MEN_CLUSTER_TAG_IDX] == 't' ]
        print(pred_obj_clusters[mention])
    return pred_obj_clusters, cluster_pred_outputs

def choose_by_env(mention, span_idxs, suffix_map, words, env_size=3):
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
            renv = clean_text(renv)
            lenv = clean_text(lenv)
            span = clean_text(span)
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
    print(mention[MEN_SPAN_STR_IDX])
    print(chosen)
    print('====')
    return chosen

CORES_SPECIAL_TOKENS = ['[[u]]', '[[f]]', '[[t]]', '<<', '>>']
REPLACE_NO_SPACE = re.compile("[`.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(\s{2,})")
def clean_text(text, special_tokens=None):
    if special_tokens:
        for tok in special_tokens:
            text = text.replace(tok, '')
    text = text.strip()
    text = REPLACE_NO_SPACE.sub("", text) 
    text = REPLACE_WITH_SPACE.sub("", text)
    text = text.strip()
    text = text.lower()
    return text

def match_mention(mention, words, words_str, suffix_map):
    i = -1
    span_str = mention[MEN_SPAN_STR_IDX].lower()
    found_idxs = []
    while i < len(words_str):
        span_idx = words_str.find(span_str, i + 1, len(words_str))

        if span_idx == -1:
            break;

        # if we match part of other words, continue
        if span_idx > 0 and words_str[span_idx - 1] != ' ':
            i = span_idx
            continue
        if span_idx + len(span_str) < (len(words_str) - 1) and words_str[span_idx + len(span_str)] != ' ':
            i = span_idx
            continue

        found_idxs.append(span_idx)
        i = span_idx
    return found_idxs

def match_mention_to_word(mention ,words):
    clean_words_str = ' '.join([w for w in words if w])
    suffix_map = create_suffix_map(words, clean_words_str)
    span_idxs = match_mention(mention, words, clean_words_str, suffix_map)
    if len(span_idxs) == 0:
        start = -1
    elif len(span_idxs) == 1:
        start = span_idxs[0]
        try:
            start = suffix_map[start]
        except:
            import ipdb; ipdb.set_trace()
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

def create_seperate_clusters(pred_obj_clusters_golden_idxs):
    clusters = [set(list(items) + [key]) for key, items in pred_obj_clusters_golden_idxs.items()]
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

def predict_final_clusters(pred_obj_clusters, words):
    unmatched_mentions = []
    pred_obj_clusters_golden_idxs = {}
    words = [clean_text(w) for w in words]
    for main_mention in pred_obj_clusters.keys():
        # clean text
        clean_main_mention = (clean_text(main_mention[MEN_LENV_IDX]), clean_text(main_mention[MEN_RENV_IDX]), clean_text(main_mention[MEN_SPAN_STR_IDX]),
                              main_mention[MEN_SPAN_RANGE_IDX], main_mention[MEN_CLUSTER_TAG_IDX])

        match_result = match_mention_to_word(clean_main_mention, words)
        if match_result is None:
            print(f'=====================')
            print(f'Could not find mention')
            print(clean_main_mention)
            print(f'=====================')
            unmatched_mentions.append(clean_main_mention)
            continue
        start, end = match_result
        pred_obj_clusters_golden_idxs[(start, end)] = []
        main_key = (start, end)

        for mention in pred_obj_clusters[main_mention]:
            clean_mention = (clean_text(mention[MEN_LENV_IDX]), clean_text(mention[MEN_RENV_IDX]), clean_text(mention[MEN_SPAN_STR_IDX]),
                             mention[MEN_SPAN_RANGE_IDX], mention[MEN_CLUSTER_TAG_IDX])
            match_result = match_mention_to_word(clean_mention, words)
            if match_result is None:
                print(f'=====================')
                print(f'Could not find mention')
                print(clean_mention)
                print(f'=====================')
                unmatched_mentions.append(clean_mention)
                continue
            start, end = match_result
            pred_obj_clusters_golden_idxs[main_key].append((start, end))

    final_pred_clusters = create_seperate_clusters(pred_obj_clusters_golden_idxs)
    clean_words_str = ' '.join([w for w in words if w])
    return final_pred_clusters, unmatched_mentions, clean_words_str


def monitor_inference(builder, tokenizer, model, model_type, config, beam_size):
    results = []
    proj_dir = r'.'
    infer_main_dir = os.path.join(proj_dir, 'inference_results')
    infer_dir = os.path.join(infer_main_dir, model_type, config, f'beam_{beam_size}')
    try:
        os.system(f'mkdir -p {infer_dir}')
    except:
        pass

    print(f'Inference Directory: {infer_dir}')
    done_keys = []
    doc_keys = list(builder.original_examples.keys())
    for current_doc_key in doc_keys:
        try:
            current_doc_key_dirname = current_doc_key.replace('/', '#')
            doc_key_dir = os.path.join(infer_dir, current_doc_key_dirname)
            if os.path.isdir(doc_key_dir):
                meta_path = os.path.join(doc_key_dir, 'meta.json')
                paragraphs_count = None
                with open(meta_path, 'rb') as f:
                    json_str = f.read().decode('ascii')
                    meta = json.loads(json_str)
                    paragraphs_count = meta['paragraphs_count'] 

                if paragraphs_count is None:
                    continue

                files = os.listdir(doc_key_dir)
                files = [f for f in files if 'paragraph' in f]
                if len(files) == paragraphs_count:
                    done_keys.append(current_doc_key)
        except:
            continue

    ratio = 100 * len(done_keys) / len(doc_keys)
    print('Monitor Results:')
    print(f'Done keys: {done_keys}')
    print(f'Done {len(done_keys)} / {len(doc_keys)} = {ratio}%')
    print()
    return done_keys


def print_results(final_pred_clusters, golden_clusters, words, unmatched_mentions):
    print('=======================')
    words_str = ' '.join(words)
    print(words_str)
    print('=======================')
    print('Predicted:')
    for cluster in final_pred_clusters:
        values_str = tuple([' '.join(words[start : end + 1]) for start, end in cluster])
        print(f'{values_str}')
    print()
    print('Golden:')
    for cluster in golden_clusters:
        values_str = tuple([' '.join(words[start : end + 1]) for start, end in cluster])
        print(f'{values_str}')
    print()
    print(f'Unmached Mentions: {len(unmatched_mentions)}')
    for mention in unmatched_mentions:
        print(mention)
    print('=======================\n')

def process_doc_key_examples(doc_key_dir, current_doc_key, builder, tokenizer, model, model_type, config, beam_size, dummy=False):
    lock_file = os.path.join(doc_key_dir, 'lock')
    #if os.path.exists(lock_file):
    #    print(f'{current_doc_key} is locked')
    #    return

    # lock
    #with open(lock_file, 'wb') as f:
    #    f.write('locked'.encode('ascii'))

    cur_paragraph_examples = [(idx, doc_key, paragraph_id, new_words, new_clusters, new_speakers, new_conll_lines, index_shift) \
                               for (idx, doc_key, paragraph_id, new_words, new_clusters, new_speakers, new_conll_lines, index_shift) \
                               in builder.paragraph_examples if doc_key == current_doc_key]
    print(f'Infer {current_doc_key} : {doc_key_dir}')
    meta_json = {'doc_key' : current_doc_key, 
                 'paragraphs_count' : len(cur_paragraph_examples),
                 'paragraphs' : [ item[2] for item in cur_paragraph_examples]}
    meta_path = os.path.join(doc_key_dir, 'meta.json')
    with open(meta_path, 'wb') as f:
        f.write(json.dumps(meta_json).encode('ascii'))

    cur_paragraph_examples.sort(key=lambda x : x[2])
    for i, (_, doc_key, paragraph_id, sentences, golden_clusters, _, _, _) in enumerate(cur_paragraph_examples):
        print()
        print(f'Try Infering {doc_key} : {paragraph_id}')
        words = flatten_list_of_lists(sentences)
        words = [w.lower() for w in words]
        try:
            words_str = ' '.join(words)
        except UnicodeEncodeError:
            print('Unicode is not supported')
            continue

        try:
            input_words_str_md5 = hashlib.md5(words_str.encode('ascii')).hexdigest()
        except:
            input_words_str_md5 = hashlib.md5(words_str.encode('utf-8')).hexdigest()

        results_path = os.path.join(doc_key_dir, f'paragraph_{paragraph_id}.pkl')
        if os.path.exists(results_path):
            try:
                with open(results_path, 'rb') as f:
                    results = pickle.load(f)
                    pred_obj_clusters, cluster_pred_outputs, final_pred_clusters, pickled_input_words_str_md5, unmatched_mentions, words, clean_words_str = results
                    if pickled_input_words_str_md5 == input_words_str_md5:
                        print(f'Loaded {doc_key} : {paragraph_id}')
                        print_results(final_pred_clusters, golden_clusters, words, unmatched_mentions)
                        continue
            except:
                pass

        print(f'Infering {doc_key} : {paragraph_id}')
        pred_obj_clusters, cluster_pred_outputs = inference_example(model, tokenizer, words, beam_size)
        final_pred_clusters, unmatched_mentions, clean_words_str = predict_final_clusters(pred_obj_clusters, words)

        with open(results_path, 'wb') as f:
            results = (pred_obj_clusters, cluster_pred_outputs, final_pred_clusters, input_words_str_md5, unmatched_mentions, words, clean_words_str)
            pickle.dump(results, f)
            print(f'Saved {doc_key} : {paragraph_id} - {results_path}')

        print_results(final_pred_clusters, golden_clusters, words, unmatched_mentions)
    
    print(f'Finished Infereing {current_doc_key}')
    # unlock
    try:
        os.remove(lock_file)
    except:
        pass
    return


def generate_inference_results(builder, tokenizer, model, model_type, config, beam_size, done_keys):
    results = []
    proj_dir = r'.'
    infer_main_dir = os.path.join(proj_dir, 'inference_results')
    infer_dir = os.path.join(infer_main_dir, model_type, config, f'beam_{beam_size}')
    try:
        os.system(f'mkdir -p {infer_dir}')
    except:
        pass

    # first, update all keys that already have a dir.
    doc_keys = list(builder.original_examples.keys())
    doc_keys = list(set(doc_keys) - set(done_keys))
    #doc_keys.shuffle()
    updated_keys = []
    for doc_key_dirname in os.listdir(infer_dir):
        current_doc_key = doc_key_dirname.replace('#', '/')
        if not current_doc_key in doc_keys:
            print(f'{current_doc_key} dont exist!')
        doc_key_dir = os.path.join(infer_dir, doc_key_dirname)
        if os.path.isdir(doc_key_dir):
            process_doc_key_examples(doc_key_dir, current_doc_key, builder, tokenizer, model, model_type, config, beam_size)
            updated_keys.append(current_doc_key)
        else:
            print(f'{doc_key_dir} is not a directory!') 

    ratio = 100 * len(updated_keys) / len(doc_keys)
    print(f'Updated keys: {len(updated_keys)} / {len(doc_keys)} = {ratio}%')
    # process keys who dont have a directory
    for current_doc_key in doc_keys:
        if current_doc_key in updated_keys:
            print(f'{current_doc_key} has already updated')
            continue

        current_doc_key_dirname = current_doc_key.replace('/', '#')
        doc_key_dir = os.path.join(infer_dir, current_doc_key_dirname)
        if not os.path.isdir(doc_key_dir):
            os.mkdir(doc_key_dir)
            print(f'Create {current_doc_key} dir: {doc_key_dir}')
        else:
            print(f'Dir exists {current_doc_key} : {doc_key_dir}')
        process_doc_key_examples(doc_key_dir, current_doc_key, builder, tokenizer, model, model_type, config, beam_size)
        updated_keys.append(current_doc_key)

def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--model', type=str)
    parser.add_argument('--builder', type=str)
    parser.add_argument('--beam', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--monitor', type=bool, default=False)
    parser.add_argument('--dummy', type=bool, default=False)
    args = parser.parse_args(sys.argv[1:])
    config = f'{args.dropout}'

    builder, tokenizer, model = load_pickles(args.model, args.builder, args.beam, config)
    done_keys = monitor_inference(builder, tokenizer, model, args.model, config, args.beam)
    if args.monitor:
        while True:
            time.sleep(60)
            done_keys = monitor_inference(builder, tokenizer, model, args.model, config, args.beam)
    else:
        done_keys = []
        generate_inference_results(builder, tokenizer, model, args.model, config, args.beam, done_keys)
if __name__ == '__main__':
    main()
