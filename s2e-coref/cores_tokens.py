import json
import logging
import os

import torch
STARTING_TOKEN = '<<'
ENDING_TOKEN = '>>'
IN_CLUSTER_TOKEN = '[[T]]'
NOT_IN_CLUSTER_TOKEN = '[[F]]'
def get_cores_tokens():
    cores_tokens  = [STARTING_TOKEN, ENDING_TOKEN] # starting ending of a mention
    cores_tokens += [IN_CLUSTER_TOKEN, NOT_IN_CLUSTER_TOKEN] # whether mentino is inside the cluster or not. TODO: with and without F token
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
                if cluster_tag is not None:
                    if cluster_index == cluster_tag:
                        sentence[mention[1]] += ' ' + IN_CLUSTER_TOKEN
                    else:
                        sentence[mention[1]] += ' ' + NOT_IN_CLUSTER_TOKEN
    return sentence, ' '.join(sentence)


def decode(sentence):
    delete_indexes = []
    sentence = sentence.split(' ')
    import ipdb; ipdb.set_trace()
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
    sentence = [w for w in sentence if w]

    start_tokens = [(i, STARTING_TOKEN, None) for i,w in enumerate(sentence) if STARTING_TOKEN in w]
    end_tokens = [(i, ENDING_TOKEN, None) for i,w in enumerate(sentence) if ENDING_TOKEN in w and IN_CLUSTER_TOKEN not in w and NOT_IN_CLUSTER_TOKEN not in w]
    end_tokens += [(i, ENDING_TOKEN, True) for i,w in enumerate(sentence) if ENDING_TOKEN in w and IN_CLUSTER_TOKEN in w]
    end_tokens += [(i, ENDING_TOKEN, False) for i,w in enumerate(sentence) if ENDING_TOKEN in w and NOT_IN_CLUSTER_TOKEN in w]
    spanning_tokens = start_tokens + end_tokens 
    spanning_tokens.sort(key=lambda x:x[0])
    missing_endings = []
    mentions_stack = []
    mentions = []
    for i, tok, c_tag in spanning_tokens:
        if STARTING_TOKEN == tok:
            mentions_stack.append((i, tok, c_tag))
        if ENDING_TOKEN == tok:
            if mentions_stack:
                s_i, _ , _ = mentions_stack.pop() 
                mentions.append(((s_i, i), c_tag))
            else:
                missing_endings.append((i, tok, c_tag))

    missing_tokens = mentions_stack + missing_endings
    missing_tokens.sort(key=lambda x:x[0])
    clusters = { True : [], False : [] }
    textual_clusters = { True : [], False : [] }
    textual_mentions = []
    for m, c_tag in mentions:
        textual_mention = ' '.join(sentence[m[0] : m[1] + 1])
        for tok in [STARTING_TOKEN, ENDING_TOKEN, IN_CLUSTER_TOKEN, NOT_IN_CLUSTER_TOKEN]:
            textual_mention = textual_mention.replace(tok, '')
        textual_mention = "".join(textual_mention.rstrip().lstrip())
        clusters[c_tag].append(m) 
        textual_mentions.append(textual_mention) 
        textual_clusters[c_tag].append(textual_mention) 
    decode_results = { 
                       'sentence' : sentence, 
                       'mentions' : mentions, 
                       'missing_tokens' : missing_tokens,
                       'clusters' : clusters,
                       'textual_mentions' : textual_mentions,
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
    d = decode(sentence)
