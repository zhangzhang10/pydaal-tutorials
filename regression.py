
""" Classes for Linear Regression and Ridge Regression """

__author__ = 'Zhang Zhang'
__email__ = 'zhang.zhang@intel.com'

from daal.algorithms.linear_regression import training as lr_training
from daal.algorithms.linear_regression import prediction as lr_prediction
from daal.algorithms.ridge_regression import training as ridge_training
from daal.algorithms.ridge_regression import prediction as ridge_prediction
from daal.data_management import HomogenNumericTable

import numpy as np

import warnings

class LR:

    _method = 'normEq'

    def __init__(self, method = 'normEq'):
        """Initialize class parameters

        Args:
           method: The default method is based on Normal Equation ('normEq'). It
           can also be QR method ('qr')
        """

        if method != 'normEq' and method != 'qr':
            warnings.warn(method + 
            ' method is not supported. Default method is used', 
            UserWarning)

        self._method = method



    def train(self, data, responses):
        """Train a Linear Regression model.

        Args:
            data: Training data
            responses: Known responses to the training data

        Returns:
            A Linear Regression model object
        """

        # Create a training algorithm object
        if self._method == 'qr': 
            lr_training_alg = lr_training.Batch_Float64QrDense() 
        else:
            lr_training_alg = lr_training.Batch_Float64NormEqDense() 
        # Set input
        lr_training_alg.input.set(lr_training.data, data)
        lr_training_alg.input.set(lr_training.dependentVariables, responses)
        # Compute
        results = lr_training_alg.compute()
        # Return the trained model
        return results.get(lr_training.model)



    def predict(self, model, testdata, intercept = True):
        """Make prediction for unseen data using a trained model

        Args:
            model: A trained model
            testdata: New data
            intercept: A boolean to inidicate if intercept needs to be computed 

        Returns:
            A NumericTable containing predicted responses 
        """

        # Create a prediction algorithm object
        lr_prediction_alg = lr_prediction.Batch_Float64DefaultDense()
        # Set input
        lr_prediction_alg.input.setModel(lr_prediction.model, model)
        lr_prediction_alg.input.setTable(lr_prediction.data, testdata)
        # Set parameters
        lr_prediction_alg.parameter.interceptFlag = intercept
        # Compute
        results = lr_prediction_alg.compute()
        return results.get(lr_prediction.prediction)



class Ridge:

    def __init__(self):
        pass



    def train(self, data, responses, alpha = 1.0):
        """Train a Ridge Regression model.

        Args:
           data: Training data
           responses: Known responses to the training data
           alpha: Regularization parameter, a small positive value with default
           1.0

        Returns:
            A Ridge Regression model object
        """

        # Create a training algorithm object
        ridge_training_alg = ridge_training.Batch_Float64DefaultDense() 
        # Set input
        ridge_training_alg.input.set(ridge_training.data, data)
        ridge_training_alg.input.set(ridge_training.dependentVariables, responses)
        # Set parameter
        alpha_nt = HomogenNumericTable(np.array([alpha], ndmin=2))
        ridge_training_alg.parameter.ridgeParameters = alpha_nt
        # Compute
        results = ridge_training_alg.compute()
        # Return the trained model
        return results.get(ridge_training.model)



    def predict(self, model, testdata, intercept = True):
        """Make prediction for unseen data using a trained model

        Args:
            model: A trained model
            testdata: New data
            intercept: A boolean to inidicate if intercept needs to be computed 

        Returns:
            A NumericTable containing predicted responses 
        """

        # Create a prediction algorithm object
        ridge_prediction_alg = ridge_prediction.Batch_Float64DefaultDense()
        # Set input
        ridge_prediction_alg.input.setModelInput(ridge_prediction.model, model)
        ridge_prediction_alg.input.setNumericTableInput(ridge_prediction.data, testdata)
        # Set parameters
        ridge_prediction_alg.parameter.interceptFlag = intercept
        # Compute
        results = ridge_prediction_alg.compute()
        return results.get(ridge_prediction.prediction)

