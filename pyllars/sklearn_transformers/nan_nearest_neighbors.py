import logging
logger = logging.getLogger(__name__)

import functools
import numpy as np
import scipy.spatial.distance

import networkx as nx
import sklearn.neighbors
from pyllars.validation_utils import check_is_fitted

import pyllars.math_utils as math_utils
from pyllars.sklearn_transformers.nan_standard_scaler import NaNStandardScaler

class NanNearestNeighbors(object):
    """ A simple kNN implementation which handles missing data
    
    Specifically, this approach uses the "Sparse-KNN" method proposed in
        Liu et al., Toxicological Sciences, 2012, vol. 129, pp. 57-73.
        
    N.B. We cannot use any of the sklearn interfaces because they always
    check for np.nans.
        
    Parameters
    ----------
    metric: callable which takes two feature vectors as input
        The base metric to use for calculating distance. In general, this
        function cannot make assumptions about the interpretation of specific
        indices unless prior knowledge is available about missingness patterns
        in the data. Please see `math_utils.distance_with_nans` for more
        details.
        
    n_neighbors: int
        The number of neighbors to consider
        
    scale: bool
        Whether to scale the features before calculating distances. The scaling
        is robust to missing values (it just ignores them), but it is only
        meaningful for numeric features.
        
    normalize_distance: bool
        Whether to normalize the distance calculated by `metric` by the number
        of shared, observed features. For example, such normalization makes
        sense for a Euclidean distance-like measure, but it is not sensible for
        something like cosine distance.
    """
    def __init__(self, 
            metric=scipy.spatial.distance.euclidean,
            n_neighbors=1,
            scale=True,
            normalize_distance=True):
        
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.scale = scale
        self.normalize_distance = normalize_distance
        
    def fit(self, X, *_, **__):
        """ Construct the internal data structures for finding neighbors
        
        Parameters
        ----------
        X: data matrix
            Missing values are represented using np.nan.
            
        Returns
        -------
        self
        """
        
        if self.scale:
            self.scaler_ = NaNStandardScaler()
            X = self.scaler_.fit_transform(X)
            
        self.knn_metric_ = functools.partial(
            math_utils.distance_with_nans,
            metric=self.metric,
            normalize=self.normalize_distance
        )
        
        # we will need this later when querying
        self.X = X
        
        distance_matrix = scipy.spatial.distance.pdist(X,
            metric=self.knn_metric_)

        # convert the infs to very large numbers
        distance_matrix = scipy.spatial.distance.squareform(distance_matrix)
        distance_matrix = np.nan_to_num(distance_matrix)
        
        knn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric="precomputed"
        )
        self.knn_ = knn.fit(distance_matrix)
        
        return self
        
    def kneighbors_graph(self, n_neighbors=None, as_nx=True):
        """ Build the k-nearest neighbors graph for the training data
        
        Please see the `sklearn` documentation for more details of the
        semantics of this method:

            http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
            
        Parameters
        ----------
        n_neighbors: int
            The number of neighbors. Default: the value passed to the constructor
            
        as_nx: bool
            Whether to return the graph as a networkx Graph data structure
            (`True`) or a scipy.sparse_matrix (`False`).
            
        Returns
        -------
        kneighbors_graph: graph
            The k-nearest neighbors graph. Please see the documentation
            referenced above for more details.
        """
        check_is_fitted(self, ["knn_"])        
        
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
            
        kneighbors_graph = self.knn_.kneighbors_graph(n_neighbors=n_neighbors)
        
        if as_nx:
            kneighbors_graph = nx.from_scipy_sparse_matrix(kneighbors_graph)
            
        return kneighbors_graph
    
    def kneighbors(self, X, n_neighbors=None, return_distance=False,
            as_np=False):
        """ Find the k nearest neighbor of each instance in `X`
        
        If specified in the constructor, then the data will be scaled (using
        the respective parameters learned from the training data).
        
        N.B. This method is not implemented especially efficiently.
        
        Parameters
        ----------
        X: data matrix
            Missing values are represented using np.nan
            
        n_neighbors: int
            The number of neighbors. Default: the value passed to the
            constructor

        return_distances: bool
            Whether to return the distances to the neighbors (`True`) or not
            (`False`)
            
        as_np: bool
            Whether to return the neighbors as an `np.array` (`True`) or a list
            of lists (`False`)
            
        Returns
        -------
        distance: np.array
            The distances to the neighbors. This is only present if
            `return_distance` is `True`.

        neighbors: np.array or list of lists
            The indices of the nearest neighbors of each entity in `X` from the
            original training set.
        """
        
        check_is_fitted(self, ["knn_"])        
        
        # check if we need to scale the data
        if self.scale:
            X = self.scaler_.transform(X)

        # ensure X is the correct shape
        X = np.atleast_2d(X)
        
        # first, find the distance from each query point to each indexed point
        
        # we need n_query rows and n_indexed columns
        
        # a bit confusing, so use clearer variable names
        queries = X
        indexed = self.X
        distance_matrix = scipy.spatial.distance.cdist(queries, indexed,
            metric=self.knn_metric_)
        
        # convert the infs to very large numbers
        distance_matrix = np.nan_to_num(distance_matrix)
        
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
            
        ret = self.knn_.kneighbors(
            X=distance_matrix,
            n_neighbors=n_neighbors,
            return_distance=return_distance
        )

        # find the indices of the neighbors
        ret_indices = ret
        if return_distance:
            ret_distances = ret[0]
            ret_indices = ret[1]

        # check if we want the list of lists
        if not as_np:
            ret_indices = [
                list(ret_indices[i]) for i in range(ret_indices.shape[0])
            ]

        ret = ret_indices
        if return_distance:
            ret = (ret_distances, ret_indices)

        return ret
