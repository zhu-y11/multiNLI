import numpy as np
import re
import random
import json
import collections
import util.parameters as params
import pickle

import pdb

FIXED_PARAMETERS = params.load_parameters()

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": 0
}

PADDING = "<PAD>"
UNKNOWN = "<UNK>"

def load_nli_data(path, snli=False):
    """
    Load MultiNLI or SNLI data.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. 
    """
    data = []
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            data.append(loaded_example)
        random.seed(1)
        random.shuffle(data)
    return data

def load_nli_data_genre(path, genre, snli=True):
    """
    Load a specific genre's examples from MultiNLI, or load SNLI data and assign a "snli" genre to the examples.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. If set to true, it will overwrite the genre label for MultiNLI data.
    """
    data = []
    j = 0
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            if loaded_example["genre"] == genre:
                data.append(loaded_example)
        random.seed(1)
        random.shuffle(data)
    return data

def tokenize(string):
    string = re.sub(r'\(|\)', '', string).lower()
    return string.split()

def build_dictionary(training_datasets):
    """
    Extract vocabulary and build dictionary.
    """  
    word_counter = collections.Counter()
    for i, dataset in enumerate(training_datasets):
        for example in dataset:
            sentence1 = example['sentence1_binary_parse'] if 'sentence1_binary_parse' in example else example['sentence1_tokenized']
            sentence2 = example['sentence2_binary_parse'] if 'sentence2_binary_parse' in example else example['sentence2_tokenized']
            word_counter.update(tokenize(sentence1))
            word_counter.update(tokenize(sentence2))
        
    vocabulary = set([word for word in word_counter])
    vocabulary = list(vocabulary)
    vocabulary = [PADDING, UNKNOWN] + vocabulary
        
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))

    return word_indices

def sentences_to_padded_index_sequences(word_indices, datasets):
    """
    Annotate datasets with feature vectors. Adding right-sided padding. 
    """
    for i, dataset in enumerate(datasets):
        for example in dataset:
            if 'sentence1_binary_parse' in example.keys():
              for sentence in ['sentence1_binary_parse', 'sentence2_binary_parse']:
                  st = 'sentence1' if sentence == 'sentence1_binary_parse' else 'sentence2'
                  example[st + '_index_sequence'] = np.zeros((FIXED_PARAMETERS["seq_length"]), dtype=np.int32)
                  
                  # lowercase and tokenization
                  token_sequence = tokenize(example[sentence])

                  for i in range(FIXED_PARAMETERS["seq_length"]):
                      if i >= len(token_sequence):
                          index = word_indices[PADDING]
                      else:
                          if token_sequence[i] in word_indices:
                              index = word_indices[token_sequence[i]]
                          else:
                              index = word_indices[UNKNOWN]
                      example[st + '_index_sequence'][i] = index
            else:
              for sentence in ['sentence1_tokenized', 'sentence2_tokenized']:
                  st = 'sentence1' if sentence == 'sentence1_tokenized' else 'sentence2'
                  example[st + '_index_sequence'] = np.zeros((FIXED_PARAMETERS["seq_length"]), dtype=np.int32)
                  
                  # lowercase and tokenization
                  token_sequence = tokenize(example[sentence])

                  for i in range(FIXED_PARAMETERS["seq_length"]):
                      if i >= len(token_sequence):
                          index = word_indices[PADDING]
                      else:
                          if token_sequence[i] in word_indices:
                              index = word_indices[token_sequence[i]]
                          else:
                              index = word_indices[UNKNOWN]
                      example[st + '_index_sequence'][i] = index




def loadEmbedding_zeros(path, word_indices):
    """
    Load GloVe embeddings. Initializng OOV words to vector of zeros.
    """
    emb = np.zeros((len(word_indices), FIXED_PARAMETERS["word_embedding_dim"]), dtype='float32')
    
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if FIXED_PARAMETERS["embeddings_to_load"] != None:
                if i >= FIXED_PARAMETERS["embeddings_to_load"]:
                    break
            
            s = line.split()
            if s[0] in word_indices:
                emb[word_indices[s[0]], :] = np.asarray(s[1:])

    return emb


def loadEmbedding_rand(train_path, test_path, word_indices):
    """
    Load embeddings. Doing a random normal initialization for OOV words.
    """
    n = len(word_indices)
    m = FIXED_PARAMETERS["word_embedding_dim"]
    emb = np.empty((n, m), dtype=np.float32)

    emb[:,:] = np.random.normal(size=(n,m))

    # Explicitly assign embedding of <PAD> to be zeros.
    emb[0:2, :] = np.zeros((1, m), dtype="float32")

    for path in [train_path, test_path]:
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if FIXED_PARAMETERS["embeddings_to_load"] != None:
                    if i >= FIXED_PARAMETERS["embeddings_to_load"]:
                        break
                
                s = line.split()
                if s[0] in word_indices:
                    emb[word_indices[s[0]], :] = np.asarray(s[1:])

    return emb

