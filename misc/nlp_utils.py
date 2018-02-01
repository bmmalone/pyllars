"""
This module contains helpers for performing natural language processing
tasks. Often, it wraps operations from nltk: http://www.nltk.org/
"""

# grab some sample text
import nltk
import nltk.stem
import nltk.corpus
import string

PUNCTUATION_TABLE = str.maketrans('', '', string.punctuation)
ENGLISH_STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
SNOWBALL_STEMMER = nltk.stem.snowball.SnowballStemmer("english")

def clean_doc(doc):
    """ Clean the given string using a standard pipeline

    In particular, this function performs the following steps:

    1. Tokenize the text
    2. Convert to lower case
    3. Remove `string.punctuation` characters from all words
    4. Remove words which contain non-alphanumeric characters
    5. Remove stop words (`nltk.corpus.stopwords.words('english')`)
    6. Stem all remaining words (`nltk.stem.snowball.SnowballStemmer("english")`)
    7. Join the stemmed words back with spaces

    After this operation, the text is ready for downstream with, for example,
    the CountVectorizer from sklearn.

    Parameters
    ----------
    doc: str
        The string to clean up

    Returns
    -------
    cleaned_doc: str
        The cleaned up string, using the pipeline described above
    """
    # tokenize into words
    words = nltk.word_tokenize(doc)

    # convert to lower case
    words = [w.lower() for w in words]

    # remove punctuation from each word
    words = [w.translate(PUNCTUATION_TABLE) for w in words]

    # remove non-alphabetic words
    words = [w for w in words if w.isalpha()]

    # filter stopwords
    words = [w for w in words if not w in ENGLISH_STOP_WORDS]

    # stem
    words = [SNOWBALL_STEMMER.stem(w) for w in words]
    
    # join back
    words = ' '.join(words)
    
    return words
