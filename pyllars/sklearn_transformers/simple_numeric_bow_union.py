"""
A simple preprocessor to combine a numeric data matrix with any number of
sparse matrices representing bag-of-words fields. The main purpose of this
class is that it allows passing dense numeric matrices and sparse bag-of-word
matrices simultaneously. Thus, for downstream estimators which can accept sparse
matrices, there is never any need for the BoW data to be densified.
"""

import scipy.sparse

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer

import pyllars.validation_utils as validation_utils

def _get_simple_num_pipeline():
    imputer = Imputer()
    scaler = StandardScaler()
    
    pipeline = Pipeline([
        ('imputer', imputer),
        ('scaler', scaler)
    ])
    
    return pipeline

def _get_simple_bow_pipeline():
    tf = TfidfTransformer(use_idf=True)
    return tf

class SimpleNumericBoWUnion(object):
    """    
    Parameters
    ----------
    num_data : 2d np.array
        The data matrix
        
    bow_tokens : list-like of sparse matrices
        A list of distinct bag-of-words represented as sparse matrices. Each
        should have the same number of rows as `num_data`.
        
    num_pipeline : sklearn.Transformer
        A preprocessing pipeline for `num_data`
        
    bow_pipeline : sklearn.Transformer
        A preprocessing pipeline for `bow_tokens`    
    """
    def __init__(
            self,
            num_data,
            bow_tokens,
            num_pipeline=_get_simple_num_pipeline(),
            bow_pipeline=_get_simple_bow_pipeline()):
            
        self.num_data = num_data
        self.bow_tokens = bow_tokens
        self.num_pipeline = num_pipeline
        self.bow_pipeline = bow_pipeline
        
    def fit(self, *_, **__):
        """ Fit the respective pipelines to the data provided in the constructor
        """
        
        self.num_pipeline_ = sklearn.clone(self.num_pipeline)
        self.num_pipeline_.fit(self.num_data)
        
        self.bow_pipelines_ = [
            sklearn.clone(self.bow_pipeline).fit(nt)
                for nt in self.bow_tokens
        ]
        
        return self
        
    def transform(self, *_, **__):
        """ Transform the data provided to the constructor
        """
        validation_utils.check_is_fitted(self, ["num_pipeline_", "bow_pipelines_"])
        
        Xt_num = self.num_pipeline_.transform(self.num_data)
        Xt_num = scipy.sparse.csr_matrix(Xt_num)
        
        it = zip(self.bow_pipelines_, self.bow_tokens)
        Xt = [
            p.transform(nt) for p, nt in it
        ]
        
        Xt.append(Xt_num)
        Xt = scipy.sparse.hstack(Xt)
        
        # we generally want to index, so convert to csr
        Xt = Xt.tocsr()
        
        return Xt
        
    @classmethod
    def get_simple_numeric_bow_pipeline(
            cls,
            num_data,
            bow_tokens,
            estimator_template,
            num_pipeline_template = _get_simple_num_pipeline(),
            bow_pipeline_template = _get_simple_bow_pipeline()):
        """    
        Create an sklearn pipeline that chains a SimpleNumericBoWUnion preprocessor
        with the given estimator for the given numeric and bow data.
        
        Parameters
        ----------
        num_data : 2d np.array
            The data matrix
            
        bow_tokens : list-like of sparse matrices
            A list of distinct bag-of-words represented as sparse matrices. Each
            should have the same number of rows as `num_data`.
            
        estimator_template : sklearn.ClassifierMixin or sklearn.RegressorMixin
            The downstream estimator (in principle, this could also be a pipeline)
            
        num_pipeline_template : sklearn.Transformer
            A preprocessing pipeline for `num_data`
            
        num_pipeline_template : sklearn.Transformer
            A preprocessing pipeline for `bow_tokens`
        
        Returns
        -------
        pipeline : sklearn.Pipeline
            A pipeline which chains the SimpleNumericBoWUnion with the given
            estimator
        """

        preprocessor_template = SimpleNumericBoWUnion
        preprocessor_hyperparameters = {
            'num_pipeline': num_pipeline_template,
            'bow_pipeline': bow_pipeline_template
        }

        preprocessor = preprocessor_template(
            num_data=num_data,
            bow_tokens=bow_tokens,
            **preprocessor_hyperparameters
        )
        
        estimator = sklearn.clone(estimator_template)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('estimator', estimator)
        ])
        
        return pipeline