from daal.data_management import HomogenNumericTable
from daal.algorithms.kernel_function import linear, rbf
from svm_multi_class import *
from quality_metrics import *
from sklearn.datasets import load_digits
import numpy as np

# Fetch digits data
digits = load_digits()
nclasses = 10
nsamples = len(digits.images)

# Use half samples for model training, and the other half for testing
training_data = HomogenNumericTable(
        digits.data[:nsamples//2,:].astype(dtype=np.double, casting='safe', copy=False))
training_labels = HomogenNumericTable(
        digits.target[:nsamples//2, np.newaxis].astype(dtype=np.double, casting='safe', copy=False))
test_data = HomogenNumericTable(
        digits.data[nsamples//2:,:].astype(dtype=np.double, casting='safe', copy=False))
test_labels = HomogenNumericTable(
        digits.target[nsamples//2:, np.newaxis].astype(dtype=np.double, casting='safe', copy=False))

classifier = MulticlassSVM(nclasses)
classifier.setSVMParams(
        cachesize = 32000000,
        kernel = linear.Batch_Float64DefaultDense(),
        shrinking = True)
svm_model = classifier.train(training_data, training_labels)
predictions = classifier.predict(svm_model, test_data)

quality = QualityMetrics(test_labels, predictions, nclasses)
print('Average accuracy: {:.2f}%'.format(
    quality.getMulticlassAverageAccuracy()*100))
