# Unsupervised Entity Resolution
Match_finder 

unsupervised_entity_resolution

Given two sets of strings, match_finder is an algorithm which

searches for matches ("similiar" strings) for strings for each entry in strings_to_match among all entries in database. match_finder builds on the classic bag-of-ngrams approach to constructing and comparing frequency vectors for texts; instead of using a dictionary of words to count occurences, match_finder employs the hashing trick to reduce the dimensionality of the frequency vectors, greatly increasing efficiency without sacrificing significant performace.

searches a list of strings to find "close" matches for a given string of interest
