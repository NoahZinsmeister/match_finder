# Match Finder
Match Finder finds matches for a given list of strings among a database of arbitrary strings. The algorithm (written in Python 3) incorporates two important concepts in natural language processing: bag-of-ngrams frequency vectors and feature hashing. A technical description of what Match Finder does might be unsupervised clustering.

# Example
To understand how Match Finder works, I thought it would be helpful to go through each step of the matching process using an example corpus: Jane Austen's _Pride and Prejudice_ (_P&P_ from now on). The full text of the book, courtesy of [Project Gutenberg](https://www.gutenberg.org/), is found in `1342.txt`.

### Corpus Preparation
The first thing we need to do is break the book into small chunks; I though the most natural division would be into sentences. To do this, I run `1342_parser.py` which divides _P&P_ into chapters and then into paragraphs. Then I use the excellent pretrained sentence tokenizer from the [NLTK python module](http://www.nltk.org/) to divide each paragraph into individual sentences (the NLTK function call is `nltk.tokenize.sent_tokenize(paragraph)`). The final result is a csv file for every chapter, the rows of which are the sentences from that chapter. They are stored in `1342_chapters`.

### String Cleaning
Before we can convert our sentences to vectors, we need to do some string cleaning. For simplicity (I'm aware that this is probably not optimal), I simply remove anything that's not a letter, space or number from each sentence, and convert all letters to lowercase. Next, I delete very common words that have little meaning (i.e. stop words -- the, but, I, and, etc.). I use NLTK's list, accessed by `set(nltk.corpus.stopwords.words('english'))`. Finally, I determine translation rules that may aid in the similarity process. I change "dearest" to "dear" and "lizzy" to "elizabeth".

### Bag-of-Ngrams...
Now that the sentences are cleaned, we need to convert them into vectors so that our comparisons are mathematically grounded. An obvious and common way to do this is via a bag-of-ngrams frequency vector. The idea is simple: create a vector such that each dimension represents a unique ngram in a particular corpus. Then, for each sentence construct a (sparse) frequency vector by adding 1 to the appropriate dimension for each (not necessarily unique) ngram in the sentence. Unfortunately, there are two major downsides to this approach: it requires a passthrough of the entire corpus to construct a list of unique ngrams, and it cannot satisfactorily deal with new ngrams if more text is added to the corpus.

### ...and Feature Hashing
To address these problems, we employ feature hashing! The idea behind feature hashing is this: instead of letting each dimension of our frequency vector represent the count of a particular ngram, we hash each ngram and record the frequency of these hash values, using the hash value itself as the column index. There's a bit more to it, but for the sake of brevity, I'll simply refer this [Wikipedia entry](https://en.wikipedia.org/wiki/Feature_hashing) and this [blog post](http://blog.someben.com/2013/01/hashing-lang/) for more information on feature hashing. I also strongly recommend that you look at my implementation. The heavy lifting happens in `hashing_trick()`, found in `match_finder.py`.

### Vector Comparison
In case you've dozed off, here's a short recap of what's happened so far: we've turned every sentence of _P&P_ into a fixed-length vector using bag-of-ngrams and feature hashing. So now all that's left to do is compare sentence vectors to find matches. This part is simple; we choose a distance metric and find the vectors (from all vectors in the database) closest to the current vector of interest. In our _P&P_ example, for each chapter, I look at each sentence vector in turn and find the closest vector (by cosine and euclidean distance) among all the sentence vectors in all other chapters except the current one. To do this I ran `main.py` twice and produced the results stored in `output_cosine.csv` and `output_euclidean.csv`.

### Sample Tables and Analysis
Here are some examples from `output_cosine` and `output_euclidean`.

Cosine distance reflects vector angle similarity (and not necessarily vector length or spatial location), so it makes sense to see that the chapter_30 sentence and its match have quite different lengths (sentence length is highly correlated with vector length), but still seem intuitively similar (they have many common ngrams). The other matches also seem quite similar, which is good!

#### Cosine
| Strings to Match | Chapter of Origin | Matches  | Score (lower is better) |
| :--------------- |:------------------| :--------| :---------------------: |
| Bingley          | chapter_1         | Bingley? | 0.0                     |
| My eldest sister has been in town these three months. | chapter_30 | I could not allow myself to conceal that your sister had been in town three months last winter, that I had known it, and purposely kept it from him. | 0.5876 |
| Mrs. Bennet could hardly comprehend it. | chapter_50 | Mrs. Bennet could hardly contain herself. | 0.2857 |
| But it is so strange! | chapter_57 | How strange! | 0.0 |

Euclidean distance is a spatial similarity measure, so we should expect the lengths and components of the sentence vector matches to be quite similar. This is largely true. Also, notice that scores of 0.0 indicate perfect similarity. Though "Dear Lizzy!" and "MY DEAREST LIZZY,--" do not seem exactly the same, recall that before converting our sentences to vectors, we make all letters lowercase, remove all punctuation, consult the translation dictionary ("dearest" maps to "dear", "lizzy" maps to "elizabeth"), and remove stop words (including "my"). After the dust settles, we end up vectorizing the following two sentences: "dear elizabeth" and "dear elizabeth". They are perfect matches after all!

#### Euclidean
| Strings to Match | Chapter of Origin | Matches              | Score (lower is better) |
| :--------------- |:------------------| :--------------------| :---------------------: |
| Dear Lizzy!      | chapter_4         | MY DEAREST LIZZY,-- | 0.0                      |
| Which do you mean? | chapter_3 | What is it you mean? | 0.0 |
| Engaged to Mr. Collins! | chapter_22 | From Mr. Collins! | 1.7320 |
| MY DEAR SIR, | chapter_48 | Dear Sir,-- | 0.0 |
| GARDINER. | chapter_50 | GARDINER. | 0.0 |

# Usage
To use Match Finder on your own NL corpus, you will largely interact with the `match_finder()` function in `match_finder.py`. The function definition is below. It relies on the following Python modules: `os, re, csv, scipy.sparse, mmh3, math, sklearn.metrics`.

```match_finder(strings_to_match, database, stop_words = None, translation_dict = None, num_ngrams = range(1,3), dim_feature_space = 2**16, num_matches = 3, similarity_metric = 'cosine', verbose = False):```

Here are some helpful steps:

1. Figure out a consistent division of your corpus into many smaller strings.
2. Decide which strings you want to find matches for and which you want to put in the database. Pass these strings (as lists) to `match_finder()` as strings_to_match and database.
3. If you want to include stop words, pass in `set(nltk.corpus.stopwords.words('english'))` or your own list/set to stop_words.
4. If you want to translate words, pass a dictionary to translation_dict where the keys are the strings to translate and each value is the desired translation. 
5. The default ngrams to consider are unigrams (words) and bigrams. You can change this by passing an iterable or list of ints to num_ngrams.
6. The default dimensionality of the feature space is 2^16 (65,536). This number is what we mod our hash values with to regularize the length of our hashed feature vectors (see `hashing_trick()` in `match_finder.py`). If you change dim_feature_space, it should be a power of 2 (to preserve hash uniformity) and less than 2^32 (the max output of the hash function I use). 
7. Changing num_matches (up to a maximum of `len(database)`) will change how many of the closest matches in database will be returned by `match_finder()`.
8. You can change similarity_metric to one of `["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"]`.
9. If you want status updates, change verbose to True.
10. The resultant function call to `match_finder()` will return a dictionary where the keys are all original strings in strings_to_match that did not map to empty strings during string cleaning, and the values are lists of tuples where each tuple is a match, the first entry of which is the original string in database, and the second entry of which is the score (distance) between the strings.

Note: If you want to consider ngrams that don't contain only lowercase letters and numbers, you must change the string cleaning section of `match_finder()` and the `find_ngrams()` function in `match_finder.py`.
