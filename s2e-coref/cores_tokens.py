import json
import random
import logging
import os
import pickle
import time, threading, sys

import datasets
from datasets import Dataset, concatenate_datasets
from datasets import Dataset, load_metric
from utils import extract_mentions_to_predicted_clusters_from_clusters
from transformers import BartForConditionalGeneration, BartTokenizer
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
def get_cores_tokens():
    cores_tokens  = [STARTING_TOKEN, ENDING_TOKEN] # starting ending of a mention
    cores_tokens += [UNK_CLUSTER_TOKEN, IN_CLUSTER_TOKEN, NOT_IN_CLUSTER_TOKEN] # whether mentino is inside the cluster or not. TODO: with and without F token
    # I will tag the color after the ending token. therefore the decoder can context everything it saw. all the previous taggings and mentions the all the current mention and decide about the color.
    return cores_tokens

# Encodes mentions example output (cluster_tag = None)
# Encodes clusters example output (cluster_tag = i, mention_tag = None)
# Encodes clusters example input  (cluster_tag = i, mention_tag = (start,end))
def encode(sentence, clusters, cluster_tag=None, mention_tag=None):
    sentence = list(sentence)
    clusters = list(clusters)
    for cluster_index, cluster in enumerate(clusters):
        for mention in cluster:
            if STARTING_TOKEN not in sentence[mention[0]]:
                sentence[mention[0]] = STARTING_TOKEN + ' ' + sentence[mention[0]]
            if ENDING_TOKEN not in sentence[mention[1]]:
                sentence[mention[1]] =  sentence[mention[1]] + ' ' + ENDING_TOKEN
                if cluster_tag is None:
                    sentence[mention[1]] += ' ' + UNK_CLUSTER_TOKEN
                else:
                    if mention_tag is None:
                        if cluster_index == cluster_tag:
                            sentence[mention[1]] += ' ' + IN_CLUSTER_TOKEN
                        else:
                            sentence[mention[1]] += ' ' + NOT_IN_CLUSTER_TOKEN
                    else:
                        if cluster_index == cluster_tag and mention == mention_tag:
                            sentence[mention[1]] += ' ' + IN_CLUSTER_TOKEN
                        else:
                            sentence[mention[1]] += ' ' + UNK_CLUSTER_TOKEN
    return sentence # by returning the list with the tokens inside we keep the words indexes


def decode(sentence):
    delete_indexes = []
    for word_index, word in enumerate(sentence):
        if (STARTING_TOKEN == word.strip()) and (word_index < len(sentence) - 1):
            sentence[word_index] = ''
            sentence[word_index + 1] = STARTING_TOKEN + ' ' + sentence[word_index + 1]
        if (ENDING_TOKEN == word.strip()) and (word_index > 0):
            sentence[word_index] = ''
            sentence[word_index - 1] = sentence[word_index - 1] + ' ' + ENDING_TOKEN
    sentence = [w for w in sentence if w]

    for word_index, word in enumerate(sentence):
        if (IN_CLUSTER_TOKEN == word.strip()) and (word_index > 0):
            sentence[word_index] = ''
            sentence[word_index - 1] = sentence[word_index - 1] + ' ' + IN_CLUSTER_TOKEN
        if (NOT_IN_CLUSTER_TOKEN == word.strip()) and (word_index > 0):
            sentence[word_index] = ''
            sentence[word_index - 1] = sentence[word_index - 1] + ' ' + NOT_IN_CLUSTER_TOKEN
        if (UNK_CLUSTER_TOKEN == word.strip()) and (word_index > 0):
            sentence[word_index] = ''
            sentence[word_index - 1] = sentence[word_index - 1] + ' ' + UNK_CLUSTER_TOKEN
    sentence = [w for w in sentence if w]

    start_tokens = [(i, STARTING_TOKEN, None) for i,w in enumerate(sentence) if STARTING_TOKEN in w]
    end_tokens  = [(i, ENDING_TOKEN, True) for i,w in enumerate(sentence) if ENDING_TOKEN in w and IN_CLUSTER_TOKEN in w]
    end_tokens += [(i, ENDING_TOKEN, False) for i,w in enumerate(sentence) if ENDING_TOKEN in w and NOT_IN_CLUSTER_TOKEN in w]
    end_tokens += [(i, ENDING_TOKEN, 'UNK') for i,w in enumerate(sentence) if ENDING_TOKEN in w and UNK_CLUSTER_TOKEN in w]
    end_tokens += [(i, ENDING_TOKEN, None) for i,w in enumerate(sentence) if ENDING_TOKEN in w \
            and IN_CLUSTER_TOKEN not in w \
            and NOT_IN_CLUSTER_TOKEN not in w \
            and UNK_CLUSTER_TOKEN not in w]
    spanning_tokens = start_tokens + end_tokens 
    spanning_tokens.sort(key=lambda x:x[0])
    missing_tokens = []
    mentions = []

    i = 0
    while i  < len(spanning_tokens):
        tok_i, tok, tok_c_tag = spanning_tokens[i]

        if STARTING_TOKEN == tok and i < len(spanning_tokens) - 1:
            next_tok_i, next_tok, next_tok_c_tag = spanning_tokens[i + 1]
            if ENDING_TOKEN == next_tok:
                mentions.append(((tok_i, next_tok_i), next_tok_c_tag))
                i += 2
                continue

        missing_tokens.append(spanning_tokens[i])
        i += 1

    textual_missing_tokens = [ sentence[i] for i, _, _ in missing_tokens]
    clusters = { True : [], False : [], 'UNK': [], None: []}
    textual_clusters = { True : [], False : [], 'UNK': [],  None : []}
    textual_mentions = []
    for m, c_tag in mentions:
        textual_mention = ' '.join(sentence[m[0] : m[1] + 1])
        for tok in [STARTING_TOKEN, ENDING_TOKEN, IN_CLUSTER_TOKEN, NOT_IN_CLUSTER_TOKEN, UNK_CLUSTER_TOKEN]:
            textual_mention = textual_mention.replace(tok, '')
        textual_mention = "".join(textual_mention.rstrip().lstrip())
        clusters[c_tag].append(m) 
        textual_mentions.append(textual_mention) 
        textual_clusters[c_tag].append(textual_mention) 

    for index, _ in enumerate(sentence):
        for tok in [STARTING_TOKEN, ENDING_TOKEN, IN_CLUSTER_TOKEN, NOT_IN_CLUSTER_TOKEN]:
            sentence[index] = sentence[index].replace(tok, '')
    sentence = ' '.join(sentence)
    decode_results = { 
                       'sentence' : sentence, 
                       'mentions' : mentions, 
                       'textual_mentions' : textual_mentions,
                       'missing_tokens' : missing_tokens,
                       'textual_missing_tokens' : textual_missing_tokens,
                       'clusters' : clusters,
                       'textual_clusters' : textual_clusters
                     }
    return decode_results

def extract_mentions(clusters):
    mentions = {}
    for c in clusters:
        for m in c:
            mentions[m] = c
    return mentions.keys()

def str_clusters(sentence, clusters):
    sentence_clusters = [ [' '.join(sentence[p[0]:p[1] + 1]) for p in cluster] for cluster in clusters ]
    sentence_clusters = [ p for c in sentence_clusters for p in c ]
    return sentence_clusters

class CoresDatasetPreProcessor(object):
    def __init__(self, training_data_path, tokenizer, max_seq_length=-1, batch_size=1, val_size=0.2, is_test=False):
        self.mention_examples = []
        self.cluster_examples = []
        self.batch_size = batch_size
        self.val_size = val_size
        self.max_seq_length = max_seq_length

        self.tokenizer = tokenizer
        print(f"Reading dataset from {training_data_path}")
        examples, self.max_mention_num, self.max_cluster_size, self.max_num_clusters = self._parse_jsonlines(training_data_path)
        self.num_mention_examples_filtered, trunced_examples = self._entity_mention_tokenize(examples)
        if is_test:
            self.env_examples = self._mentions_with_envs(trunced_examples)

        self.num_cluster_examples_filtered = self._binary_clustering_tokenize(trunced_examples)
        self.mentions_df = pd.DataFrame(self.mention_examples, columns=['idx', 'input_str', 'output_str'])
        self.clusters_df = pd.DataFrame(self.cluster_examples, columns=['idx', 'cluster_index', 'mention', 'input_str', 'output_str'])
        print(f"Mentions: {len(self.mentions_df)}")
        print(f"Clusters: {len(self.clusters_df)}")
            
    def _parse_jsonlines(self, training_data_path):
        examples = []
        max_mention_num = -1
        max_cluster_size = -1
        max_num_clusters = -1
        with open(training_data_path, 'r') as f:
            for line in f:
                d = json.loads(line.strip())
                doc_key = d["doc_key"]
                input_words = flatten_list_of_lists(d["sentences"])
                clusters = d["clusters"]
                max_mention_num = max(max_mention_num, len(flatten_list_of_lists(clusters)))
                max_cluster_size = max(max_cluster_size, max(len(cluster) for cluster in clusters) if clusters else 0)
                max_num_clusters = max(max_num_clusters, len(clusters) if clusters else 0)
                speakers = flatten_list_of_lists(d["speakers"])
                examples.append((doc_key, input_words, clusters, speakers))
        return examples, max_mention_num, max_cluster_size, max_num_clusters

    def _trunc_words(self, words, clusters, trunc_count):
        if trunc_count > 0:
            w = words[:-trunc_count]
        else:
            w = words
        new_clusters = [ [[start, end] for start, end in cluster if start < len(w) and end < len(w)] for cluster in clusters ]
        new_clusters = [ cluster for cluster in new_clusters if cluster ]
        mentions_count = sum([len(c) for c in new_clusters])
        #print(f"trunc_count = {trunc_count}, w length = {len(w)}, mentions_count = {mentions_count}")
        return w, new_clusters

    def _process_example(self, words, clusters, trunc_step=10):
        w = list(words)
        trunc_count= 0
        total_mention_count = sum([len(c) for c in clusters])
        while w:
            w, new_clusters  = self._trunc_words(list(words), clusters, trunc_count)
            words_str        = ' '.join(w)
            tokenized_input  = self.tokenizer(words_str, padding="max_length") # padding
            input_ids        = tokenized_input['input_ids']
            input_ids        = torch.tensor(input_ids).unsqueeze(0)
            input_ids_mask   = tokenized_input['attention_mask']
            input_ids_mask   = torch.tensor(input_ids_mask).unsqueeze(0)

            entity_mentions = encode(w, new_clusters, None)
            entity_mentions = ' '.join(entity_mentions)
            tokenized_output = self.tokenizer(entity_mentions, padding="max_length")
            output_ids       = tokenized_output['input_ids']
            output_ids       = torch.tensor(output_ids).unsqueeze(0)
            output_ids_mask  = tokenized_output['attention_mask']
            output_ids_mask  = torch.tensor(output_ids_mask).unsqueeze(0)

            #output_str = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if 0 < self.max_seq_length < input_ids.shape[1]:
                trunc_count += trunc_step
                continue
                
            if 0 < self.max_seq_length < output_ids.shape[1]:
                trunc_count += trunc_step
                continue
            break
            print('Done')
        return w, words_str, new_clusters, entity_mentions, trunc_count

    def _entity_mention_tokenize(self, examples):
        trunced_examples = []
        num_examples_filtered = 0
        for idx, (_, words, clusters, _) in enumerate(examples):
            chunk_id = 0
            try:
                new_words, words_str, new_clusters, entity_mentions, trunc_count = self._process_example(words, clusters)
            except:
                continue
            self.mention_examples.append((f"{idx}_{chunk_id}", words_str, entity_mentions))
            trunced_examples.append((idx, chunk_id, new_words, new_clusters))
            print(f"mention: idx = {idx} chunk_id = {chunk_id}")

            trunc_length = len(new_words)
            while trunc_length <  len(words):
                remain_words    = words[trunc_length:]
                remain_clusters = [ [[start - trunc_length , end - trunc_length] for start, end in cluster \
                                  if start >= trunc_length and end >= trunc_length] for cluster in clusters ]
                remain_clusters = [ cluster for cluster in remain_clusters if cluster ]
                new_words, words_str, new_clusters, entity_mentions, trunc_count = self._process_example(remain_words, remain_clusters)
                chunk_id += 1
                if new_clusters:
                    self.mention_examples.append((f"{idx}_{chunk_id}", words_str, entity_mentions))
                    trunced_examples.append((idx, chunk_id, new_words, new_clusters))
                    print(f"mention: idx = {idx} chunk_id = {chunk_id}")
                else:
                    print(f"IGNORED! mention: idx = {idx} chunk_id = {chunk_id}")
                trunc_length += len(new_words)
                if new_words == remain_words:
                    break

        return num_examples_filtered, trunced_examples

    def _binary_clustering_tokenize(self, examples):
        coref_examples = []
        num_examples_filtered = 0
        for (idx, chunk_id, words, clusters) in examples:
            current_cluster_examples = []
            mentions = sum([len(c) for c in clusters])
            for c_i, cluster in enumerate(clusters):
                try:
                    cluster_output_str = encode(words, clusters, cluster_tag=c_i, mention_tag=None)
                    cluster_output_str = ' '.join(cluster_output_str)
                    cluster_output   = self.tokenizer(cluster_output_str, padding="max_length")
                    output_ids       = cluster_output['input_ids']
                    output_ids       = torch.tensor(output_ids).unsqueeze(0)
                    output_ids_mask  = cluster_output['attention_mask']
                    output_ids_mask  = torch.tensor(output_ids_mask).unsqueeze(0)
                except:
                    num_examples_filtered += len(cluster)
                    continue
                    
                if 0 < self.max_seq_length < output_ids.shape[1]:
                    num_examples_filtered += len(cluster)
                    continue

                for mention in cluster:
                    try:
                        mention_input_str   = encode(words, clusters, cluster_tag=c_i, mention_tag=mention)
                        mention_input_str   = ' '.join(mention_input_str)
                        mention_input   = self.tokenizer(mention_input_str, padding="max_length")
                        input_ids       = mention_input['input_ids']
                        input_ids       = torch.tensor(input_ids).unsqueeze(0)
                        input_ids_mask  = mention_input['attention_mask']
                        input_ids_mask  = torch.tensor(input_ids_mask).unsqueeze(0)
                    except Exception as e:
                        num_examples_filtered += 1
                        continue

                    if 0 < self.max_seq_length < input_ids.shape[1]:
                        num_examples_filtered += 1
                        continue

                    current_cluster_examples.append((f"{idx}_{chunk_id}", c_i, mention, mention_input_str, cluster_output_str))
            print(f"clusters: idx = {idx} chunk_id = {chunk_id} mention_examples = {len(current_cluster_examples)} / {mentions}")
            self.cluster_examples.extend(current_cluster_examples)
        return num_examples_filtered

    @staticmethod
    def clean_span(span):
        span = span.strip()
        EXCLUDE_TOKENS = ',.?:><'
        for token in EXCLUDE_TOKENS:
            span = span.replace(token, '')
        span = span.strip()
        return span

    def _mentions_with_envs(self, examples, env_size=4):
        examples_with_env = []
        for (idx, chunk_id, words, clusters) in examples:
            current_cluster_examples = []
            mentions = extract_mentions_to_predicted_clusters_from_clusters(clusters)
            mentions = list(mentions.keys())
            mentions_env = {}
            span_mention_lookup = {}
            for start, end in mentions:
                span = ' '.join(words[start : end + 1])
                renv = ' '.join(words[end + 1 : end + 1 + env_size])
                lenv = ' '.join(words[max(0, start - env_size) : start])
                mentions_env[(start, end)] = {'span' : span, 'renv' : renv, 'lenv' : lenv}
                clean_span = CoresDatasetPreProcessor.clean_span(span)
                if clean_span not in span_mention_lookup:
                    span_mention_lookup[clean_span] = []
                span_mention_lookup[clean_span].append({'span' : span, 'span_idxs' : (start, end), 'renv' : renv, 'lenv' : lenv})
            examples_with_env.append((idx, chunk_id, words, clusters, mentions_env, span_mention_lookup))

        return examples_with_env

    def __len__(self):
        return len(self.examples)

WordsExample = ['--', 'basically', ',', 'it', 'was', 'unanimously', 'agreed', 'upon', 'by', 'the', 'various', 'relevant', 'parties', '.', 'To', 'express', 'its', 'determination', ',', 'the', 'Chinese', 'securities', 'regulatory', 'department', 'compares', 'this', 'stock', 'reform', 'to', 'a', 'die', 'that', 'has', 'been', 'cast', '.', 'It', 'takes', 'time', 'to', 'prove', 'whether', 'the', 'stock', 'reform', 'can', 'really', 'meet', 'expectations', ',', 'and', 'whether', 'any', 'deviations', 'that', 'arise', 'during', 'the', 'stock', 'reform', 'can', 'be', 'promptly', 'corrected', '.', 'Dear', 'viewers', ',', 'the', 'China', 'News', 'program', 'will', 'end', 'here', '.', 'This', 'is', 'Xu', 'Li', '.', 'Thank', 'you', 'everyone', 'for', 'watching', '.', 'Coming', 'up', 'is', 'the', 'Focus', 'Today', 'program', 'hosted', 'by', 'Wang', 'Shilin', '.', 'Good-bye', ',', 'dear', 'viewers', '.']

ClusterExample = [[[16, 16], [19, 23]], [[42, 44], [57, 59], [25, 27]], [[83, 83], [82, 82]]]
def test_encoder_decoder():
    print(str_clusters(W,C))
    for c_i, c in enumerate(C):
        print()
        print(c_i)
        sentence = encode(W,C,c_i)
        print(sentence)
        print(str_clusters(sentence, C))
        print()
        d = decode(sentence)
        s1 = sentence + ['<<', 'Hello']
        d = decode(s1)
        print(d['missing_tokens'])
        print(d['textual_missing_tokens'])
        print()
        s2 = sentence + ['Hello', '>>', '[[T]]']
        d = decode(s2)
        print(d['missing_tokens'])
        print(d['textual_missing_tokens'])
        print()
        s3 = sentence + ['<<', 'Hello', '>>']
        d = decode(s3)
        print(d['clusters'])
        print(d['textual_clusters'])
        print()
        s4 = [c for c in sentence if IN_CLUSTER_TOKEN not in c and NOT_IN_CLUSTER_TOKEN in c]
        d = decode(s4)
        print(d['clusters'])
        print(d['textual_clusters'])
        print()
        s5 = [c for c in sentence if STARTING_TOKEN not in c]
        d = decode(s5)
        print(d['missing_tokens'])
        print(d['textual_missing_tokens'])
        print(d['clusters'])
        print(d['textual_clusters'])

    s6= encode(W,C, None)
    d = decode(s6)
    print(d['missing_tokens'])
    print(d['textual_missing_tokens'])
    print(d['clusters'])
    print(d['textual_clusters'])
    print()

def create_datasets():
    # path
    model_type = sys.argv[1]
    if model_type not in ('bert', 't5', 'bart'):
        print('Invalid Model Type')
        sys.exit(0)

    training_data_path = sys.argv[2]
    if not os.path.exists(training_data_path):
        print(f'Path dont exists {training_data_path}')
        sys.exit(0)

    print("Loading tokenizer")
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

    filename = os.path.basename(training_data_path)
    dataset_builder_path = os.path.join('.', 'builders', f'{filename}.builder.{model_type}.pkl')
    print(f'Builder path: {dataset_builder_path}')

    if os.path.exists(dataset_builder_path):
        with open(dataset_builder_path, 'rb') as f:
            builder = pickle.load(f)
    else:
        builder = CoresDatasetPreProcessor(training_data_path, tokenizer, max_seq_length=128)
        with open(dataset_builder_path, 'wb') as f:
            pickle.dump(builder, f)
            print(f"Success: {dataset_builder_path}")

if __name__ == '__main__':
    create_datasets()
