# Match Finder
Match Finder finds matches for a given list of strings among a database of arbitrary strings. The algorithm incorporates two important concepts in natural language processing: bag-of-ngrams frequency vectors and feature hashing. A technical description of what this algorithm does might be something like unsupervised entity resolution. The algorithm solves an unsupervised learning problem because it matches unlabeled strings, and it helps resolve entities by finding different strings that may refer to the same concept or idea (the same entity).

# Function

```python
def match_finder(strings_to_match, database, stop_words = None,
                 translation_dict = None, num_ngrams = range(1,3),
                 dim_feature_space = 2**16, num_matches = 3,
                 similarity_metric = 'cosine', verbose = False):
    """
    For each string in strings_to_match, match_finder finds close
    matches ("similiar" strings) among the strings in database.

    Match_finder builds on the classic bag-of-ngrams approach to
    constructing and comparing frequency vectors for texts; instead of
    using a dictionary of words to count occurences, match_finder
    employs the hashing trick to reduce the dimensionality of the
    frequency vectors, greatly increasing efficiency without sacrificing
    significant performace.

    variable descriptions:
    strings_to_match: a list of strings to be matched

    database: a list of strings to search for matches among

    stop_words: list or set of strings to ignore

    translation_dict: dict of strings and values to translate

    num_ngrams: iter or list of how many ngrams to consider

    dim_feature_space: dimension of hashed feature space

    num_matches: int indicating how many of the closest matches to save

    similarity_metric: string of metric used to compare feature rows,
    one of ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"]

    verbose: one of [True, False], prints percentage progress reports
    """
```

# Example
To understand how Match Finder works, I thought it would be helpful to go through each step of the algorithm using an example corpus: Jane Austen's _Pride and Prejudice_ (_P&P_ from now on). The full text of the book, courtesy of [Project Gutenberg](https://www.gutenberg.org/), is found in `1342.txt`.

### Corpus Preparation
The first thing we need to do is break the book into small chunks that we can compare; I though the most natural division would be into sentences. To do this, I ran `1342_parser.py` which first divides _P&P_ into chapters, then into paragraphs, and then uses the excellent pretrained sentence tokenizer from the [NLTK python module](http://www.nltk.org/) to finally divide each chapter into individual sentences. The NLTK function call is `nltk.tokenize.sent_tokenize(paragraph)`. What we end up with is a csv file for every chapter (all in `1342_chapters`) where the rows are all the sentences from that chapter.

### String Cleaning
Before we can convert our sentences to vectors, we need to do some string cleaning. For simplicity (I'm aware that this is probably not optimal), I simply remove anything that's not a letter, space or number from each sentence, and convert all letters to lowercase.

### Bag-of-Ngrams...

Now that we've cleaned our sentences, we need to convert them into a vector so that our comparisons can be consistent and mathematical. An obvious and common way to do this is via a bag-of-ngrams frequency vector. The idea is simple: create a really long vector where each dimension represents a unique ngram in a particular corpus, and for each sentence construct a (sparse) frequency vector by adding all sentence ngram counts to the appropriate dimension. Unfortunately, there are two major downsides to this approach: it requires a passthrough of the entire corpus to construct a list of unique ngrams, and it cannot satisfactorily deal with new ngrams if more text is added to the corpus.

### ...and Feature Hashing
To address these problems, we employ feature hashing! The idea behind feature hashing is this: instead of having each dimension of our frequency vector represent the count of a particular ngram, lets hash each ngram and record the frequency of each hash value in the column index of the hash value. There's a bit more to it, but for the sake of brevity, I'll simply mention this [Wikipedia entry](https://en.wikipedia.org/wiki/Feature_hashing) and this [blog post](http://blog.someben.com/2013/01/hashing-lang/) for more information on feature hashing. The implementation of this hashing-modified bag-of-ngrams approach is in the (well-commented) `match_finder.py`.

### Vector Comparison
In case you've dozed off, here's a short recap of what's happened so far: we've turned every sentence of _P&P_ into a fixed-length vector using bag-of-ngrams and feature hashing. So now all that's left to do is compare sentence vectors to find matches. In this example, for each chapter, I look at each sentence vector in the current chapter and find the two closest vectors (by cosine and euclidean distance) among all the sentence vectors in all other chapters except the current one. I ran `main.py` and it produced the results stored in `output_cosine.csv` and `output_euclidean.csv`.

### Sample Tables?

Cosine:

| Strings to Match | Chapter of Origin | Matches  | Score (lower is better) |
| :--------------- |:------------------| :--------| :---------------------: |
| Bingley          | chapter_1         | Bingley? | 0                       |

Euclidean:

| Strings to Match | Chapter of Origin | Matches              | Score (lower is better) |
| :--------------- |:------------------| :--------------------| :---------------------: |
| "Dear Lizzy!"    | chapter_4         | "MY DEAREST LIZZY,-- | 0                       |


# Usage
To use Match Finder with your own NL corpus, do the following:

1. Some things.
2. More things!
