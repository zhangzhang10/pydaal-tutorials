
""" A class for K-Means clustering """

__author__ = 'Zhang Zhang'
__email__ = 'zhang.zhang@intel.com'

import daal.algorithms.kmeans as kmeans
from daal.algorithms.kmeans import init
from daal.data_management import HomogenNumericTable

import numpy as np


class KMeans:

    def __init__(self, nclusters, randomseed = None):
        """Initialize class parameters
        
        Args:
           nclusters: Number of clusters
           randomseed: An integer used to seed the random number generator
        """

        self.nclusters_ = nclusters
        self.seed_ = 1234 if randomseed is None else randomseed
        self.centroids_ = None
        self.assignments_ = None
        self.goalfunction_ = None
        self.niterations_ = None


    def compute(self, data, centroids = None, maxiters = 100):
        """Compute K-Means clustering for the input data

        Args:
           data: Input data to be clustered
           centroids: User defined input centroids. If None then initial
               centroids will be randomly chosen
           maxiters: The maximum number of iterations
        """

        if centroids is not None:
            # Create an algorithm object for centroids initialization
            init_alg = init.Batch_Float64RandomDense(self.nclusters_)
            # Set input
            init_alg.input.set(init.data, data)
            # Set parameters
            init_alg.parameter.seed = self.seed_
            # Compute initial centroids
            self.centroids_ = init_alg.compute().get(init.centroids)
        else:
            self.centroids_ = centroids

        # Create an algorithm object for clustering
        clustering_alg = kmeans.Batch_Float64LloydDense(
                self.nclusters_,
                maxiters)
        # Set input
        clustering_alg.input.set(kmeans.data, data)
        clustering_alg.input.set(kmeans.inputCentroids, self.centroids_)
        # compute
        result = clustering_alg.compute()
        self.centroids_ = result.get(kmeans.centroids)
        self.assignments_ = result.get(kmeans.assignments)
        self.goalfunction_ = result.get(kmeans.goalFunction)
        self.niterations_ = result.get(kmeans.nIterations)


