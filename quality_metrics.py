
""" A class for two-class and multi-class classifier quality metrics"""

__author__ = 'Zhang Zhang'
__email__ = 'zhang.zhang@intel.com'

from daal.algorithms.multi_class_classifier import quality_metric_set as multiclass_quality
from daal.algorithms.classifier.quality_metric import multiclass_confusion_matrix 
from daal.algorithms.svm import quality_metric_set as twoclass_quality
from daal.algorithms.classifier.quality_metric import binary_confusion_matrix
from daal.data_management import BlockDescriptor_Float64, readOnly

class QualityMetrics:

    _truth = None
    _predictions = None
    _twoclass_metrics = None
    _multiclass_metrics = None

    def __init__(self, truth, predictions, nclasses = 2):
        """Initialize class parameters

        Args:
           truth: ground truth
           predictions: predicted labels
           nclasses: number of classes
        """

        self._truth = truth
        self._predictions = predictions
        if nclasses == 2:
            self._computeTwoclassQualityMetrics()
        elif nclasses > 2:
            self._computeMulticlassQualityMetrics(nclasses)
        else:
            raise ValueError('nclasses must be at least 2')


    def getTwoclassAccuracy(self):
        """Return accuray from two-class classification quality metrics
        """
        return self._twoclass_metrics[binary_confusion_matrix.accuracy]

    def getTwoclassPrecision(self):
        """Return precision from two-class classification quality metrics
        """
        return self._twoclass_metrics[binary_confusion_matrix.precision]

    def getTwoclassRecall(self):
        """Return recall from two-class classification quality metrics
        """
        return self._twoclass_metrics[binary_confusion_matrix.recall]

    def getTwoclassFscore(self):
        """Return fscore from two-class classification quality metrics
        """
        return self._twoclass_metrics[binary_confusion_matrix.fscore]

    def getTwoclassSpecificity(self):
        """Return specificity from two-class classification quality metrics
        """
        return self._twoclass_metrics[binary_confusion_matrix.specificity]

    def getTwoclassAUC(self):
        """Return AUC from two-class classification quality metrics
        """
        return self._twoclass_metrics[binary_confusion_matrix.AUC]


    def getMulticlassAverageAccuracy(self):
        """Return average accuracy from multi-class classification quality metrics
        """
        return self._multiclass_metrics[multiclass_confusion_matrix.averageAccuracy]

    def getMulticlassErrorRate(self):
        """Return error rate from multi-class classification quality metrics
        """
        return self._multiclass_metrics[multiclass_confusion_matrix.errorRate]

    def getMulticlassMicroPrecision(self):
        """Return micro precision from multi-class classification quality metrics
        """
        return self._multiclass_metrics[multiclass_confusion_matrix.microPrecision]

    def getMulticlassMicroRecall(self):
        """Return micro recall from multi-class classification quality metrics
        """
        return self._multiclass_metrics[multiclass_confusion_matrix.microRecall]

    def getMulticlassMicroFscore(self):
        """Return micro fscore from multi-class classification quality metrics
        """
        return self._multiclass_metrics[multiclass_confusion_matrix.microFscore]

    def getMulticlassMacroPrecision(self):
        """Return macro precision from multi-class classification quality metrics
        """
        return self._multiclass_metrics[multiclass_confusion_matrix.macroPrecision]

    def getMulticlassMacroRecall(self):
        """Return macro recall from multi-class classification quality metrics
        """
        return self._multiclass_metrics[multiclass_confusion_matrix.macroRecall]

    def getMulticlassMacroFscore(self):
        """Return macro fscore from multi-class classification quality metrics
        """
        return self._multiclass_metrics[multiclass_confusion_matrix.macroFscore]


    def _computeTwoclassQualityMetrics(self):
        # Alg object for quality metrics computation
        quality_alg = twoclass_quality.Batch()
        # Get access to the input parameter
        input = quality_alg.getInputDataCollection().getInput(
                twoclass_quality.confusionMatrix)
        # Pass ground truth and predictions as input
        input.set(binary_confusion_matrix.groundTruthLabels, self._truth)
        input.set(binary_confusion_matrix.predictedLabels, self._predictions)
        # Compute confusion matrix
        confusion = quality_alg.compute().getResult(twoclass_quality.confusionMatrix)
        # Retrieve quality metrics from the confusion matrix
        metrics = confusion.get(binary_confusion_matrix.binaryMetrics)
        # Convert the metrics into an ndarray and return it
        block = BlockDescriptor_Float64()
        metrics.getBlockOfRows(0, 1, readOnly, block)
        self._twoclass_metrics = block.getArray().flatten()

    def _computeMulticlassQualityMetrics(self, nclasses):
        # Alg object for quality metrics computation
        quality_alg = multiclass_quality.Batch(nclasses)
        # Get access to the input parameter
        input = quality_alg.getInputDataCollection().getInput(
                multiclass_quality.confusionMatrix)
        # Pass ground truth and predictions as input
        input.set(multiclass_confusion_matrix.groundTruthLabels, self._truth)
        input.set(multiclass_confusion_matrix.predictedLabels, self._predictions)
        # Compute confusion matrix
        confusion = quality_alg.compute().getResult(multiclass_quality.confusionMatrix)
        # Retrieve quality metrics from the confusion matrix
        metrics = confusion.get(multiclass_confusion_matrix.multiClassMetrics)
        # Convert the metrics into an ndarray and return it
        block = BlockDescriptor_Float64()
        metrics.getBlockOfRows(0, 1, readOnly, block)
        self._multiclass_metrics = block.getArray().flatten()


    
