import nltk
import os
import csv
from match_finder import match_finder

if __name__ == "__main__":
    # reads stop words and replacement dict into memory
    stop_words = set(nltk.corpus.stopwords.words('english'))
    translation_dict = {}
    with open(os.getcwd() + "/translation_rules.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for i in range(1, len(row)):
                translation_dict[row[i]] = row[0]
    
    # creates a dictionary of chapters
    chapters = {}
    for i in range(61):
        sentence_list = []
        chapter = "chapter_" + str(i+1)
        with open(os.getcwd() + "/1342_chapters/" + chapter + ".csv", "r") \
            as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                sentence_list.append(row[0])
            chapters[chapter] = sentence_list

    stop_words = stop_words
    translation_dict = translation_dict
    num_ngrams = range(1,5)
    dim_feature_space = 2**16
    num_matches = 1
    similarity_metric = "euclidean"
    verbose = True

    matches_to_write = {}
    for chapter in chapters:
        # search for matches for each sentence in a given chapter
        strings_to_match = chapters[chapter]
        database = []
        # searches among all sentences in every other chapter
        for other_chapter in chapters:
            if other_chapter != chapter:
                database += chapters[other_chapter]

        matches = match_finder(strings_to_match,
                               database,
                               stop_words = stop_words,
                               translation_dict = translation_dict,
                               num_ngrams = num_ngrams,
                               dim_feature_space = dim_feature_space,
                               num_matches = num_matches,
                               similarity_metric = similarity_metric,
                               verbose = verbose)

        # sorts strings by the distance of closest match
        strings_to_match_sorted = sorted(matches, key = lambda x:
                                         matches[x][0][1])
        # gets corresponding matches for each string
        matched_from_database = [matches[x] for x in strings_to_match_sorted]
        # saves the top 2 matches from this chapter
        matches_to_write[chapter] = (strings_to_match_sorted[0:2],
                                     matched_from_database[0:2])

    with open("output_" + similarity_metric + ".csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        # write header
        header = ["Strings to Match", "Chapter of Origin", "Matches",
                  "Score (lower is better)"]
        writer.writerow(header)
        # write matches
        for match in sorted(matches_to_write,
                            key = lambda x: int(x.split("_")[1])):
            for i, row in enumerate(matches_to_write[match][0]):
                row_write = []
                row_write.append(row)
                row_write.append(match)
                # write original match text
                row_write.append(matches_to_write[match][1][i][0][0])
                # write match score
                row_write.append(matches_to_write[match][1][i][0][1])
                writer.writerow(row_write)
