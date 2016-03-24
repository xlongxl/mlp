import sys
sys.path.append('../lib')

import numpy as np
import mlp

from sklearn import cross_validation
from sklearn import preprocessing

# http://archive.ics.uci.edu/ml/datasets/Iris
fname = 'data/iris.csv'

n_instances = 150

n_attributes = 4
attributes_cols = range(4)

n_classes = 3
classes_col = 4

examples = np.genfromtxt(fname, delimiter=',', usecols=attributes_cols)
examples = preprocessing.MinMaxScaler().fit_transform(examples)

classes = np.genfromtxt(fname, dtype=None, delimiter=',', usecols=classes_col)
class_dict = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}

targets = np.zeros((n_instances, n_classes))
for class_, target in zip(classes, targets):
    target[class_dict[class_]] = 1.0

nn = mlp.MLP([n_attributes + 1, 6, n_classes], debug=False)

kfolds = cross_validation.KFold(n_instances, n_folds=5, shuffle=True)
for train_indices, test_indices in kfolds:
    train_examples = examples[train_indices]
    train_targets = targets[train_indices]

    nn.fit(train_examples, train_targets, n_epochs=2000, learning_rate=0.1)

    test_examples = examples[test_indices]
    test_targets = targets[test_indices]

    count = 0.0
    for example, target in zip(test_examples, test_targets):
        count += target[nn.predict(example)]
    print "accuracy:", count / len(test_targets)
