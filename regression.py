
""" Classes for Linear Regression and Ridge Regression """

__author__ = 'Zhang Zhang'
__email__ = 'zhang.zhang@intel.com'

from daal.algorithms.linear_regression import training as lr_training
from daal.algorithms.linear_regression import prediction as lr_prediction
from daal.algorithms.ridge_regression import training as ridge_training
from daal.algorithms.ridge_regression import prediction as ridge_prediction
from daal.data_management import HomogenNumericTable

from utils import *

import numpy as np


def getBetas(linear_model):
    """Return regression coefficients for a given linear model

    Args:
        linear_model: A trained model

    Returns:
        A n-by-(k+1) NumericTable contains betas, where n is the number of dependent
        variables; k is the number of features (independent variables)
    """

    return linear_model.getBeta()




def mse(values, fitted_values):
    """Return Mean Squared Errors for fitted values w.r.t. true values

    Args:
        values: True values. NumericTable, nsamples-by-noutputs
        fitted_values: True values. NumericTable, nsamples-by-noutputs

    Returns:
        A tuple contains MSE's
    """

    y_t = getArrayFromNT(values)
    y_p = getArrayFromNT(fitted_values)
    rss = ((y_t - y_p) ** 2).sum(axis = 0)
    mse = rss / y_t.shape[0]
    return tuple(mse)




def score(y_true, y_pred):
    """Compute R-squared and adjusted R-squared

    Args:
        y_true: True values. NumericTable, shape = (nsamples, noutputs)
        y_pred: Predicted values. NumericTable, shape = (nsamples, noutputs)

    Returns:
        R2: A tuple with noutputs values
    """

    y_t = getArrayFromNT(y_true)
    y_p = getArrayFromNT(y_pred)
    rss = ((y_t - y_p) ** 2).sum(axis = 0)
    tss = ((y_t - y_t.mean(axis = 0)) ** 2).sum(axis = 0)
    return (1 - rss/tss)



class LinearRegression:


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

        self.method_ = method



    def train(self, data, responses):
        """Train a Linear Regression model.

        Args:
            data: Training data
            responses: Known responses to the training data

        Returns:
            A Linear Regression model object
        """

        # Create a training algorithm object
        if self.method_ == 'qr': 
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



