from daal.data_management import HomogenNumericTable
from daal.algorithms.kernel_function import linear, rbf
from svm_multi_class import *
from quality_metrics import *
import numpy as np

datahome = './mldata'

# Load training data from file optdigits.tra
tra = np.genfromtxt(datahome+'/optdigits.tra', dtype = np.double, delimiter=',')
# Split the data into training data and labels, and create numeric tables
training_data = HomogenNumericTable(tra[:, :64])
training_labels = HomogenNumericTable(tra[:, 64:])

# Load testing data from file optdigits.tes
tes = np.genfromtxt(datahome+'/optdigits.tes', dtype = np.double, delimiter=',')
# Split the data into testing data and labels, and create numeric tables
test_data = HomogenNumericTable(tes[:, :64])
test_labels = HomogenNumericTable(tes[:, 64:])

nclasses = 10
classifier = MulticlassSVM(nclasses)
classifier.setSVMParams(
        shrinking = True,
        kernel = rbf.Batch())
svm_model = classifier.train(training_data, training_labels)
predictions = classifier.predict(svm_model, test_data)

quality = QualityMetrics(test_labels, predictions, nclasses)
print('Average accuracy: {:.2f}%'.format(
    quality.getMulticlassAverageAccuracy()*100))
