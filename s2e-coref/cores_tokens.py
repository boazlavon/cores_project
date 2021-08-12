import json
import logging
import os

import torch

STARTING_TOKEN = '<<'
ENDING_TOKEN = '>>'
UNK_CLUSTER_TOKEN = '[[U]]'
IN_CLUSTER_TOKEN = '[[T]]'
NOT_IN_CLUSTER_TOKEN = '[[F]]'
def get_cores_tokens():
    cores_tokens  = [STARTING_TOKEN, ENDING_TOKEN] # starting ending of a mention
    cores_tokens += [UNK_CLUSTER_TOKEN, IN_CLUSTER_TOKEN, NOT_IN_CLUSTER_TOKEN] # whether mentino is inside the cluster or not. TODO: with and without F token
    # I will tag the color after the ending token. therefore the decoder can context everything it saw. all the previous taggings and mentions the all the current mention and decide about the color.
    return cores_tokens

W = ['--', 'basically', ',', 'it', 'was', 'unanimously', 'agreed', 'upon', 'by', 'the', 'various', 'relevant', 'parties', '.', 'To', 'express', 'its', 'determination', ',', 'the', 'Chinese', 'securities', 'regulatory', 'department', 'compares', 'this', 'stock', 'reform', 'to', 'a', 'die', 'that', 'has', 'been', 'cast', '.', 'It', 'takes', 'time', 'to', 'prove', 'whether', 'the', 'stock', 'reform', 'can', 'really', 'meet', 'expectations', ',', 'and', 'whether', 'any', 'deviations', 'that', 'arise', 'during', 'the', 'stock', 'reform', 'can', 'be', 'promptly', 'corrected', '.', 'Dear', 'viewers', ',', 'the', 'China', 'News', 'program', 'will', 'end', 'here', '.', 'This', 'is', 'Xu', 'Li', '.', 'Thank', 'you', 'everyone', 'for', 'watching', '.', 'Coming', 'up', 'is', 'the', 'Focus', 'Today', 'program', 'hosted', 'by', 'Wang', 'Shilin', '.', 'Good-bye', ',', 'dear', 'viewers', '.']

C = [[[16, 16], [19, 23]], [[42, 44], [57, 59], [25, 27]], [[83, 83], [82, 82]]]
def encode(sentence, clusters, cluster_tag=None):
    sentence = list(sentence)
    clusters = list(clusters)
    for cluster_index, cluster in enumerate(clusters):
        for mention in cluster:
            if STARTING_TOKEN not in sentence[mention[0]]:
                sentence[mention[0]] = STARTING_TOKEN + ' ' + sentence[mention[0]]
            if ENDING_TOKEN not in W[mention[1]]:
                sentence[mention[1]] =  sentence[mention[1]] + ' ' + ENDING_TOKEN
                if cluster_tag is None:
                    sentence[mention[1]] += ' ' + UNK_CLUSTER_TOKEN
                else:
                    if cluster_index == cluster_tag:
                        sentence[mention[1]] += ' ' + IN_CLUSTER_TOKEN
                    else:
                        sentence[mention[1]] += ' ' + NOT_IN_CLUSTER_TOKEN
    return sentence, ' '.join(sentence)


def decode(sentence):
    delete_indexes = []
    sentence = sentence.split(' ')
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


print(str_clusters(W,C))
for c_i, c in enumerate(C):
    print()
    print(c_i)
    sentence_list, sentence = encode(W,C,c_i)
    print(sentence)
    print(str_clusters(sentence_list, C))
    print()
    d = decode(sentence)
    s1 = sentence + ' << Hello!'
    d = decode(s1)
    print(d['missing_tokens'])
    print(d['textual_missing_tokens'])
    print()
    s2 = sentence + ' Hello! >> [[T]]'
    d = decode(s2)
    print(d['missing_tokens'])
    print(d['textual_missing_tokens'])
    print()
    s3 = sentence + ' << Hello! >>'
    d = decode(s3)
    print(d['clusters'])
    print(d['textual_clusters'])
    print()
    s4 = sentence.replace(IN_CLUSTER_TOKEN, '')
    s4 = s4.replace(NOT_IN_CLUSTER_TOKEN, '')
    d = decode(s4)
    print(d['clusters'])
    print(d['textual_clusters'])
    print()
    s5 = sentence.replace(STARTING_TOKEN, '')
    d = decode(s5)
    print(d['missing_tokens'])
    print(d['textual_missing_tokens'])
    print(d['clusters'])
    print(d['textual_clusters'])

_, s6= encode(W,C, None)
d = decode(s6)
print(d['missing_tokens'])
print(d['textual_missing_tokens'])
print(d['clusters'])
print(d['textual_clusters'])
print()
