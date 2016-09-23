
""" A class for multi-class classifier based on SVM algorithm"""

__author__ = 'Zhang Zhang'
__email__ = 'zhang.zhang@intel.com'

from daal.algorithms.svm import training as svm_training
from daal.algorithms.svm import prediction as svm_prediction
from daal.algorithms.kernel_function import linear, rbf
from daal.algorithms.multi_class_classifier import training as multiclass_training
from daal.algorithms.multi_class_classifier import prediction as multiclass_prediction
from daal.algorithms.classifier import training as training_params
from daal.algorithms.classifier import prediction as prediction_params

class MulticlassSVM:

    _nclasses = 0
    _svm_training_alg = None
    _svm_prediction_alg = None

    def __init__(self, nclasses):
        """Initialize class parameters

        Args:
           nclasses: number of classes
        """

        self._nclasses = nclasses
        # Create an SVM two-class classifier object for training
        self._svm_training_alg = svm_training.Batch_Float64DefaultDense()
        # Create an SVM two-class classifier object for prediction
        self._svm_prediction_alg = svm_prediction.Batch_Float64DefaultDense()

    
    def setSVMParams(self, 
            cachesize = 1000000000, 
            C = 1.0,
            sigma = 1.0,
            kernel = linear.Batch_Float64DefaultDense(),
            shrinking = False):
        """Tweak SVM training and prediction algorithm parameters

        Args:
            cachesize: size of chache in bytes for storing kernel matrix
            kernel: SVM kernel, can be either linear or rbf
            sigma: Coefficient of the rbf kernel
            shrinking: whether do shrinking optimization or not
        """

        self._svm_training_alg.parameter.cacheSize = cachesize 
        self._svm_training_alg.parameter.C = C 
        if getattr(kernel.parameter, 'sigma', None):
            kernel.parameter.sigma = sigma
        self._svm_training_alg.parameter.kernel = kernel
        self._svm_prediction_alg.parameter.kernel = kernel
        self._svm_training_alg.parameter.doShrinking = shrinking



    def train(self, data, labels):
        """Train an SVM model.

        Args:
            data: training data
            labels: ground truth known for training data 

        Returns:
            An SVM model object
        """
        
        # Create a multiclass classifier object based on the
        # SVM two-class classifier
        multiclass_training_alg = multiclass_training.Batch_Float64OneAgainstOne()
        multiclass_training_alg.parameter.nClasses = self._nclasses
        multiclass_training_alg.parameter.training = self._svm_training_alg
        multiclass_training_alg.parameter.prediction = self._svm_prediction_alg

        # Pass training data and labels
        multiclass_training_alg.input.set(training_params.data, data)
        multiclass_training_alg.input.set(training_params.labels, labels)

        # Build the model and return it
        return multiclass_training_alg.compute().get(training_params.model)


    def predict(self, model, testdata):
        """Make predictions for unseen data using a learned model.

        Args:
            model: a learned SVM model
            testdata: new data

        Returns:
            A NumericTable containing predicted labels
        """

        # Create a multiclass classifier object based on the
        # SVM two-class classifier
        multiclass_prediction_alg = multiclass_prediction.Batch_Float64DefaultDenseOneAgainstOne()
        multiclass_prediction_alg.parameter.nClasses = self._nclasses
        multiclass_prediction_alg.parameter.training = self._svm_training_alg
        multiclass_prediction_alg.parameter.prediction = self._svm_prediction_alg
        
        # Pass a model and input data
        multiclass_prediction_alg.input.setModel(prediction_params.model, model)
        multiclass_prediction_alg.input.setTable(prediction_params.data, testdata)

        # Return prediction results
        results = multiclass_prediction_alg.compute()
        return results.get(prediction_params.prediction)

