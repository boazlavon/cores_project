import json
import logging
import os
import pickle
from collections import namedtuple

import torch
import pandas as pd

from consts import SPEAKER_START, SPEAKER_END, NULL_ID_FOR_COREF
from utils import flatten_list_of_lists
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoTokenizer, CONFIG_MAPPING, LongformerConfig, BertConfig, BertTokenizer
from transformers import BertGenerationConfig, BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, EncoderDecoderConfig

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


CondCoresExample = namedtuple("CondCoresExample", ["input_ids", "input_attention_mask", "output_ids"])
class CondCoresDataset(Dataset):
    def __init__(self, file_path, tokenizer, mentions_or_clusters, max_seq_length=-1, model=None):
        self.mentions_examples = []
        self.mentions_df = None
        self.clusters_examples = []
        self.clusters_df = None

        self.mentions_or_clusters = mentions_or_clusters
        self.device = torch.device('cuda')
        if model is not None:
            self.model = model.to(self.device)
        self.tokenizer = tokenizer
        print(f"Reading dataset from {file_path}")
        examples, self.max_mention_num, self.max_cluster_size, self.max_num_clusters = self._parse_jsonlines(file_path)
        self.max_seq_length = max_seq_length
        if self.mentions_or_clusters == 'mentions':
            self.examples, self.num_examples_filtered = self._entity_mention_tokenize(examples)
        elif self.mentions_or_clusters == 'clusters':
            self.examples, self.num_examples_filtered = self._binary_clustering_tokenize(examples)
        else:
            raise ValueError("Must choose mentions/clusters for mentions_or_clusters")
        print(
            f"Finished preprocessing Coref dataset. {len(self.examples)} examples were extracted, {self.num_examples_filtered} were filtered due to sequence length.")
        self._to_pandas()

    def _to_pandas(self):
        self.mentions_df = pd.DataFrame(self.mentions_examples, columns=['input_str', 'output_str'])
        self.clusters_df = pd.DataFrame(self.clusters_examples, columns=['input_ids', 'input_mask', 'output_ids'])

    def _parse_jsonlines(self, file_path):
        examples = []
        max_mention_num = -1
        max_cluster_size = -1
        max_num_clusters = -1
        with open(file_path, 'r') as f:
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

    def _execute(self, input_ids, output_ids):
        input_ids = input_ids.to(self.device)
        output_ids = output_ids.to(self.device)
        loss = self.model(input_ids=input_ids, decoder_input_ids=output_ids, labels=output_ids)
        print("Loss: {}".format(str(loss)))
        loss[0].backward()
        print("Loss backword")
        
    def _entity_mention_tokenize(self, examples):
        coref_examples = []
        num_examples_filtered = 0
        for idx, (_, words, clusters, _) in enumerate(examples):
            try:
                words_str        = ' '.join(words)
                tokenized_input  = self.tokenizer(words_str, padding="max_length") # padding
                input_ids        = tokenized_input['input_ids']
                input_ids        = torch.tensor(input_ids).unsqueeze(0)
                input_ids_mask   = tokenized_input['attention_mask']
                input_ids_mask   = torch.tensor(input_ids_mask).unsqueeze(0)

                entity_mentions = encode(words, clusters, None)
                entity_mentions = ' '.join(entity_mentions)
                tokenized_output = self.tokenizer(entity_mentions, padding="max_length")
                output_ids       = tokenized_output['input_ids']
                output_ids       = torch.tensor(output_ids).unsqueeze(0)
                output_ids_mask  = tokenized_output['attention_mask']
                output_ids_mask  = torch.tensor(output_ids_mask).unsqueeze(0)
            except Exception as e:
                continue

            #output_str = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if 0 < self.max_seq_length < input_ids.shape[1]:
                num_examples_filtered += 1
                continue

            if 0 < self.max_seq_length < output_ids.shape[1]:
                num_examples_filtered += 1
                continue

            print(f"mention: idx = {idx}")
            coref_examples.append(CondCoresExample(input_ids=input_ids, input_attention_mask=input_ids_mask, output_ids=output_ids))
            self.mentions_examples.append((words_str, entity_mentions))
        return coref_examples, num_examples_filtered

    def _binary_clustering_tokenize(self, examples):
        coref_examples = []
        num_examples_filtered = 0
        for idx, (_, words, clusters, _) in enumerate(examples):
            current_example_clusters = []
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

                    current_example_clusters.append(CondCoresExample(input_ids=input_ids, input_attention_mask=input_ids_mask, output_ids=output_ids))
                    #self.clusters_examples.append((mention_input_str, cluster_output_str))
                    self.clusters_examples.append((input_ids, input_ids_mask, output_ids))
            print(f"clusters: idx = {idx} mentions_examples = {len(current_example_clusters)} /  {mentions}")
            coref_examples.extend(current_example_clusters)
        return coref_examples, num_examples_filtered

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

def test_encoder_decoder():
    W = ['--', 'basically', ',', 'it', 'was', 'unanimously', 'agreed', 'upon', 'by', 'the', 'various', 'relevant', 'parties', '.', 'To', 'express', 'its', 'determination', ',', 'the', 'Chinese', 'securities', 'regulatory', 'department', 'compares', 'this', 'stock', 'reform', 'to', 'a', 'die', 'that', 'has', 'been', 'cast', '.', 'It', 'takes', 'time', 'to', 'prove', 'whether', 'the', 'stock', 'reform', 'can', 'really', 'meet', 'expectations', ',', 'and', 'whether', 'any', 'deviations', 'that', 'arise', 'during', 'the', 'stock', 'reform', 'can', 'be', 'promptly', 'corrected', '.', 'Dear', 'viewers', ',', 'the', 'China', 'News', 'program', 'will', 'end', 'here', '.', 'This', 'is', 'Xu', 'Li', '.', 'Thank', 'you', 'everyone', 'for', 'watching', '.', 'Coming', 'up', 'is', 'the', 'Focus', 'Today', 'program', 'hosted', 'by', 'Wang', 'Shilin', '.', 'Good-bye', ',', 'dear', 'viewers', '.']

    C = [[[16, 16], [19, 23]], [[42, 44], [57, 59], [25, 27]], [[83, 83], [82, 82]]]
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

def process_data_to_model_inputs(batch):
    import ipdb; ipdb.set_trace()
    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]
    return batch

def create_datasets():
    # path
    proj_dir = r'/home/yandex/AMNLP2021/boazlavon/cores_project/s2e-coref'
    proj_dir = r'/home/yandex/AMNLP2021/boazlavon/cores_project/s2e-coref'
    data_dir = os.path.join(proj_dir, 'bert2bert_coref_data')
    cache_dir = os.path.join(proj_dir, 'bert2bert_cache')
    #file_path = os.path.join(data_dir, 'train.english.jsonlines')
    file_path = os.path.join(data_dir, 'mini_train.english.jsonlines')
    tokenizer_path = os.path.join(proj_dir, 'bert2bert_model', 'tokenizer')
    model_path = os.path.join(proj_dir, 'bert2bert_model', 'model')

    print("Loading pre-trained tokenizer")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, cache_dir=cache_dir)
    print("pre-trained: tokenizer: {}".format(len(tokenizer)))
    clusters_path = os.path.join(data_dir, 'clusters.pkl')
    if os.path.exists(clusters_path):
        with open(clusters_path, 'rb') as f:
            clusters_db = pickle.load(f)
    else:
        clusters_db  = CondCoresDataset(file_path, tokenizer, 'clusters', max_seq_length=512)
        with open(clusters_path, 'wb') as f:
            pickle.dump(clusters_db, f)

    mentions_path = os.path.join(data_dir, 'mentions.pkl')
    if os.path.exists(mentions_path):
        with open(mentions_path, 'rb') as f:
            mentions_db = pickle.load(f)
    else:
        mentions_db = CondCoresDataset(file_path, tokenizer, 'mentions', max_seq_length=512)
        with open(mentions_path, 'wb') as f:
            pickle.dump(mentions_db, f)
    import ipdb; ipdb.set_trace()

create_datasets()
