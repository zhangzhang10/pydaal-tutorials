
""" A class for classifier quality metrics."""

__author__ = 'Zhang Zhang'
__email__ = 'zhang.zhang@intel.com'

from daal.algorithms.multi_class_classifier import quality_metric_set as multiclass_quality
from daal.algorithms.classifier.quality_metric import multiclass_confusion_matrix 
from daal.algorithms.svm import quality_metric_set as twoclass_quality
from daal.algorithms.classifier.quality_metric import binary_confusion_matrix
from daal.data_management import BlockDescriptor_Float64, readOnly

from collections import namedtuple


# Two-class quality metrics type
TwoClassMetrics = namedtuple('TwoClassMetrics',
        ['accuracy', 'precision', 'recall', 'fscore', 'specificity', 'auc'])

# Multi-class quality metrics type
MultiClassMetrics = namedtuple('MultiClassMetrics',
        ['accuracy', 'error_rate', 'micro_precision', 'micro_recall',
         'micro_fscore', 'macro_precision', 'macro_recall', 'macro_fscore'])


class ClassifierQualityMetrics:


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



    def get(self, metric):
        """Get a metric from the quality metrics collection

        Args:
           metric: name of the metric to return

        Returns:
           A numeric value for the given metric
        """

        return getattr(self._metrics, metric)



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
        # Convert the metrics into a Python namedtuple and return it
        block = BlockDescriptor_Float64()
        metrics.getBlockOfRows(0, 1, readOnly, block)
        x = block.getArray().flatten()
        self._metrics = TwoClassMetrics(*x)
        metrics.releaseBlockOfRows(block)



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
        # Convert the metrics into a Python namedtuple and return it
        block = BlockDescriptor_Float64()
        metrics.getBlockOfRows(0, 1, readOnly, block)
        x = block.getArray().flatten()
        self._metrics = MultiClassMetrics(*x)
        metrics.releaseBlockOfRows(block)


