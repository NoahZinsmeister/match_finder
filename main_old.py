import os
import csv
from resolver_test import match_finder

if __name__ == "__main__":
    # reads stop words and replacement dict into memory
    stop_words = set()
    with open(os.getcwd() + "/stop_words.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            stop_words.add(row[0])
    translation_dict = {}
    with open(os.getcwd() + "/translation_rules.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for i in range(1, len(row)):
                translation_dict[row[i]] = row[0]
    
    # reads the two string corpora into memory
    corpora_1 = []
    with open(os.getcwd() + "/sample_strings_to_match.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            corpora_1.append(row[0])
    corpora_2 = []
    with open(os.getcwd() + "/sample_database.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            corpora_2.append(row[0])
    
    strings_to_match = corpora_1
    database = corpora_2
    stop_words = stop_words
    translation_dict = translation_dict
    num_ngrams = range(1,5)
    dim_feature_space = 2**16
    num_matches = 2
    similarity_metric = "cosine"
    verbose = True

    matches = match_finder(strings_to_match,
                           database,
                           stop_words = stop_words,
                           translation_dict = translation_dict,
                           num_ngrams = num_ngrams,
                           dim_feature_space = dim_feature_space,
                           num_matches = num_matches,
                           similarity_metric = similarity_metric,
                           verbose = verbose)

    #prints matches to a CSV, sorted by the score of the closest match
    strings_to_match_sorted = sorted(matches, key = lambda x: matches[x][0][1])
    matched_from_database = [matches[x] for x in strings_to_match_sorted]
    with open("sample_output.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        # write header
        header = ["String to Match"]
        for i in range(num_matches):
            header.append("Match " + str(i+1))
            header.append("Score " + str(i+1))
        writer.writerow(header)
        # write matches
        for i, row in enumerate(strings_to_match_sorted):
            row_clean = []
            row_clean.append(row)
            # for each of the closest num_matches matches
            for j in range(num_matches):
                # write original match text
                row_clean.append(matched_from_database[i][j][0])
                # write match score
                row_clean.append(matched_from_database[i][j][1])
            writer.writerow(row_clean)
