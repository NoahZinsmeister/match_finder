import os
import re
import csv
import scipy.sparse
import mmh3
import math
import sklearn.metrics

def find_ngrams(iterable_ints, input_string):
    """takes a sequence of ints and a string, returns list of ngrams"""
    ngrams = []
    # finds individual words
    onegrams = re.findall(r"(?=(\b[a-z0-9]+))", input_string)
    # does some indexing to find arbitrary ngrams 
    for n in iterable_ints:
        ngrams += [" ".join(onegrams[i:i+n]) for i in \
            range(0,len(onegrams)-n+1)]
    return ngrams

def hashing_trick(input_string, iterable_ints, dim_feature_space):
    """takes a string and returns a hashed ngram feature dictionary"""
    dictionary = {}
    # returns specified number of ngrams
    ngrams_in_line = find_ngrams(iterable_ints, input_string)
    """
    hashes each ngram, mods the output with dim_feature_space to make
    a list of signed 32bit ints, the absolute value of which represent
    column indices (the sign is so that the collisions are unbiased)
    """
    tokens = [mmh3.hash(ngram, 2) for ngram in ngrams_in_line]
    tokens = [math.copysign(x % dim_feature_space, x) for x in tokens]
    # adds signed count of each token to the dictionary
    for token in tokens:
        try:
            dictionary[abs(token)] += int(math.copysign(1,token))
        except KeyError:
            dictionary[abs(token)] = int(math.copysign(1,token))
    return dictionary

def dict_to_sparse(feature_dict, dim_feature_space):
    '''takes a hashing_trick() dict and returns a sparse feature row'''
    keys = list(iter(feature_dict))
    # feature_dict.values() might work but I am worried about order...
    values = list(feature_dict[x] for x in iter(feature_dict))
    num_values = len(keys)
    sparse_row = scipy.sparse.csr_matrix((values, ([0]*num_values, keys)), 
                                         (1,dim_feature_space))
    return sparse_row

def ignore(input_string, stop_words):
    """deletes words in input_string iff they appear in stop_words"""
    output = [word for word in input_string.split() if word not in stop_words]
    return " ".join(output)

def trans(input_string, translation_dict):
    """
    translates each word in input_string iff it has an entry in
    translation_dict
    """
    output = [word for word in input_string.split()]
    for i, word in enumerate(output):
        try:
            output[i] = translation_dict[word]
        except KeyError:
            pass
    return " ".join(output)

def match_finder(strings_to_match, database, stop_words = None,
                 translation_dict = None, num_ngrams = range(1,3),
                 dim_feature_space = 2**16, num_matches = 3,
                 similarity_metric = 'cosine', verbose = False):
    strings_to_match_clean = []
    database_clean = []

    # string cleaning
    for name in [strings_to_match, database]:
        for string in name:
            # convert all whitespaces to one space, converts to lowercase 
            clean_string = re.sub(r"\s+", " ", string.lower()).strip()
            # removes anything that's not a space, letter, or number
            clean_string = re.sub(r"[^a-z0-9\s]", "", clean_string)
            if name == strings_to_match:
                strings_to_match_clean.append(clean_string)
            else:
                database_clean.append(clean_string)
    
    # remove stop words
    if stop_words != None:
        for name in [strings_to_match_clean, database_clean]:
            for i, string in enumerate(name):
                if name == strings_to_match_clean:
                    strings_to_match_clean[i] = ignore(string, stop_words)
                else:
                    database_clean[i] = ignore(string, stop_words)
    
    # translate words
    if translation_dict != None:
        for name in [strings_to_match_clean, database_clean]:
            for i, string in enumerate(name):
                if name == strings_to_match_clean:
                    strings_to_match_clean[i] = trans(string, translation_dict)
                else:
                    database_clean[i] = trans(string, translation_dict)

    # construct sparse matrix of ngram features in database
    db_size = len(database_clean)
    database_sparse = scipy.sparse.csr_matrix((0,dim_feature_space))
    for i, line in enumerate(database_clean):
        # progress meter
        if verbose == True:
            if round(i/db_size, 1) != round((i+1)/db_size, 1):
                print("Making Sparse Matrix:",
                      int(round((i+1)/db_size, 1)*100), "% complete")
        row_dict = hashing_trick(line, num_ngrams, dim_feature_space)
        new_row = dict_to_sparse(row_dict, dim_feature_space)
        database_sparse = scipy.sparse.vstack([database_sparse, new_row])

    """
    for each string in strings_to_match_clean, finds the 3 most similar
    strings in database_clean (using cosine similarity) and add them
    and their score to matches
    """
    match_size = len(strings_to_match_clean)
    matches = {}
    for i, line in enumerate(strings_to_match_clean):
        # progress bar
        if verbose == True:
            if round(i/match_size, 1) != round((i+1)/match_size, 1):
                print("Matching Strings:",
                      int(round((i+1)/match_size, 1)*100), "% complete")
        # sparsify with hash trick
        txt_dict = hashing_trick(line, num_ngrams, dim_feature_space)
        sparse = dict_to_sparse(txt_dict, dim_feature_space)
        # calculate pairwise cosine distances
        distances = sklearn.metrics.pairwise.pairwise_distances(sparse, \
            database_sparse, metric = similarity_metric)
        distances = distances[0].tolist()
        # does some indexing/sorting shenanigans to find top x matches
        vals = sorted(distances)[0:num_matches]
        inds = [j for j, dist in enumerate(distances) if dist in vals]
        inds.sort(key = lambda ind: distances[ind])
        del inds[num_matches:]
        strs = [database[k] for k in inds]
        matches[strings_to_match[i]] = list(zip(strs, vals))
    return matches
