"""
This module contains helpers for performing natural language processing
tasks. Often, it wraps operations from nltk: http://www.nltk.org/
"""

import nltk
import nltk.stem
import nltk.corpus
import string

import typing
from typing import Collection, Mapping, Optional

###
# Constants
###
STRING_PUNCTUATION_TABLE = str.maketrans('', '', string.punctuation)
"""A mapping from 'punctuation' characters to `None`"""

ENGLISH_STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
"""English stopwords from `nltk`"""

ENGLISH_SNOWBALL_STEMMER = nltk.stem.snowball.SnowballStemmer("english")
"""English snowball stemmer from `nltk`"""

###
# NLP parsing
###

def clean_doc(
        doc:str,
        replacement_table:Optional[Mapping]=None,
        stop_words:Optional[Collection]=None,
        stemmer:nltk.stem.StemmerI=ENGLISH_SNOWBALL_STEMMER) -> str:
    """ Clean the given string using a standard pipeline

    In particular, this function performs the following steps:

    1. Tokenize the text
    2. Convert to lower case
    3. Replace `replacement_table` characters from all words
    4. Remove words which contain non-alphanumeric characters
    5. Remove `stop_words`
    6. Stem all remaining words using `stemmer`
    7. Join the stemmed words back with spaces

    After this operation, the text is ready for downstream with, for example,
    :class:`sklearn.feature_extraction.text.CountVectorizer`.

    Parameters
    ----------
    doc: str
        The string to clean up
        
    replacement_table : typing.Optional[typing.Mapping]
        A mapping from characters to replacement characters. For example,
        the default is a mapping from standard punctuation characters to
        `None`, which has the effect of removing those characters from `doc`.
        
        By default, :func:`pyllars.nlp_utils.STRING_PUNCTUATION_TABLE` is
        used.
        
    stop_words : typing.Optional[typing.Collection]
        A set of words to remove. Only (case-insensitive) exact matches
        are removed.
        
        By default, :func:`pyllars.nlp_utils.ENGLISH_STOP_WORDS` is
        used.
        
    stemmer : nltk.stem.StemmerI
        A stemmer.

    Returns
    -------
    cleaned_doc : str
        The cleaned up string, using the pipeline described above
    """    
    if replacement_table is None:
        replacement_table = STRING_PUNCTUATION_TABLE
        
    if stop_words is None:
        stop_words = ENGLISH_STOP_WORDS
        
    # tokenize into words
    words = nltk.word_tokenize(doc)

    # convert to lower case
    words = [w.lower() for w in words]

    # remove punctuation from each word
    words = [w.translate(replacement_table) for w in words]

    # remove non-alphabetic words
    words = [w for w in words if w.isalpha()]

    # filter stopwords
    words = [w for w in words if not w in stop_words]

    # stem
    words = [stemmer.stem(w) for w in words]
    
    # join back
    words = ' '.join(words)
    
    return words
