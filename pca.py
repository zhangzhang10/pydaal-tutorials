
""" A class for PCA using pyDAAL """

__author__ = 'Zhang Zhang'
__email__ = 'zhang.zhang@intel.com'

import daal.algorithms.pca as pca
from daal.data_management import HomogenNumericTable

import numpy as np


class PCA:

    def __init__(self, method = 'correlation'):
        """Initialize class parameters

        Args:
           method: The default method is based on correation matrix. It
           can also be the SVD method ('svd')
        """

        if method != 'correlation' and method != 'svd':
            warnings.warn(method + 
            ' method is not supported. Default method is used', 
            UserWarning)

        self.method_ = method
        self.eigenvalues_ = None
        self.eigenvectors_ = None


    def compute(self, data):
        """Compute PCA the input data

        Args:
           data: Input data 
        """

        # Create an algorithm object for PCA
        if self.method_ == 'svd':
            pca_alg = pca.Batch_Float64SvdDense()
        else:
            pca_alg = pca.Batch_Float64CorrelationDense()

        # Set input
        pca_alg.input.setDataset(pca.data, data)
        # compute
        result = pca_alg.compute()
        self.eigenvalues_ = result.get(pca.eigenvalues)
        self.eigenvectors_ = result.get(pca.eigenvectors)

