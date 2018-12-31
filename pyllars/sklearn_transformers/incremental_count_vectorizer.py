"""
The class is similar to `sklearn.feature_extraction.text.CountVectorizer`.
However, it provides move control over when pruning and other operations
are performed. Thus, it can be used to incrementally collect sufficient
statistics for a large corpus without requiring it all fit in memory.
Additionally, counting with multiple instances of this class are inherently
parallelizable. The `klass.merge` method then allows the combination of the
independent counters.
"""
import logging
logger = logging.getLogger(__name__)

import collections
import numpy as np
import scipy.sparse
from pyllars.validation_utils import check_is_fitted

import pyllars.utils as utils

class IncrementalCountVectorizer:
    def __init__(self, 
            min_df = 0,
            max_df=1,
            num_docs=None,
            prune=True,
            create_mapping=True,
            token_count=None,
            get_tokens=None):
        
        self.min_df = min_df
        self.max_df = max_df
        self.num_docs = num_docs
        self.prune = prune
        self.create_mapping = create_mapping
        self.get_tokens = get_tokens
        
        if token_count is not None:
            self.token_count_ = token_count
        
    def fit(self, X, *_, **__):
        """ Extract the counts
        
        Parameters
        ----------
        X: iterable of tokens, or something `self.get_tokens` can process
            This should either be an iterable of something which can be passed
            into the `set` constructor (such as a list of lists of string
            tokens) or something which, when `self.get_tokens` is called on
            it, produces such an iterable
            
        Returns
        -------
        self
        """ 
        self.token_count_ = collections.defaultdict(int)
        self.num_docs = len(X)

        for tokens in X:
            # transform the tokens, if necessary
            if self.get_tokens is not None:
                tokens = self.get_tokens(tokens)
            
            # now count the tokens in this set
            token_set = set(tokens)
            for token in token_set:
                self.token_count_[token] += 1
               
        # prune if necessary
        if self.prune:
            self.prune_tokens()
            
        # and create the mapping
        if self.create_mapping:
            self.create_token_mapping()
                
        return self
    
    def transform(self, X, *_, **__):
        """ Convert the iterable of tokens `X` into a (sparse) bag of words
        
        Parameters
        ----------
        X: iterable of tokens, or something `self.get_tokens` can process
            This should either be an iterable of something which can be passed
            into the `set` constructor (such as a list of lists of string
            tokens) or something which, when `self.get_tokens` is called on
            it, produces such an iterable
            
        Returns
        -------
        list of lists of token identifiers
            The list of token indices for each document in `X`
        """
        check_is_fitted(self, "token_mapping_")
        
        Xt = []
        
        for tokens in X:
            # transform the tokens, if necessary
            if self.get_tokens is not None:
                tokens = self.get_tokens(tokens)
                
            # grab the tokens we know
            int_tokens = [
                self.token_mapping_.get(t) for t in tokens
            ]
            
            # remove those we did not know
            int_tokens = utils.remove_nones(int_tokens)
            
            Xt.append(int_tokens)
            
        return Xt
    
    def create_token_mapping(self):
        self.token_mapping_ = dict()

        for i, token in enumerate(sorted(self.token_count_.keys())):
            self.token_mapping_[token] = i
                
    def prune_tokens(self):
        """ Remove rare and abundant tokens
        """
        
        if self.min_df <= 1:
            self.min_df = self.num_docs * self.min_df

        if self.max_df <= 1:
            self.max_df = self.num_docs * self.max_df
            
        #print(self.min_df, self.max_df, len(self.token_count_))        
        #print(self.token_count_)

        msg = ("[inc_count_vec.prunt] before pruning: {} tokens".format(
            len(self.token_count_)))
        logger.debug(msg)
            
        to_remove = []
        for item, value in self.token_count_.items():
            if (value < self.min_df) or (value > self.max_df):
                to_remove.append(item)

        for item in to_remove:
            self.token_count_.pop(item)
        
        msg = ("[inc_count_vec.prunt] after pruning: {} tokens".format(
            len(self.token_count_)))
        logger.debug(msg)
            
    @classmethod
    def merge(klass, vectorizers, min_df=0, max_df=1, num_docs=None, prune=True, create_mapping=True, get_tokens=None):
        """ Merge the iterable of vectorizers into a single IncrementalCountVectorizer
        """
        if num_docs is None:
            num_docs = np.sum(v.num_docs for v in vectorizers)
            
        token_count = collections.defaultdict(int)

        for v in vectorizers:
            for token, count in v.token_count_.items():
                token_count[token] += count
                
        merged_vectorizer = IncrementalCountVectorizer(
            min_df=min_df,
            max_df=max_df,
            num_docs=num_docs,
            token_count=token_count,
            get_tokens=get_tokens
        )
        
        if prune:
            merged_vectorizer.prune_tokens()
            
        if create_mapping:
            merged_vectorizer.create_token_mapping()
            
        return merged_vectorizer

           
        
        
        
