import sys
sys.path.append('../lib')

import numpy as np
import mlp

from sklearn import cross_validation
from sklearn import preprocessing

# http://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
fname = 'data/handwritten_digits.csv'

n_instances = 7494

n_attributes = 16
attributes_cols = range(16)

n_classes = 10
classes_col = 16

examples = np.genfromtxt(fname, delimiter=',', usecols=attributes_cols)
examples = preprocessing.MinMaxScaler().fit_transform(examples)

classes = np.genfromtxt(fname, delimiter=',', usecols=classes_col)

targets = np.zeros((n_instances, n_classes))
for class_, target in zip(classes, targets):
    target[class_] = 1.0

nn = mlp.MLP([n_attributes + 1, 15, n_classes], debug=True)

kfolds = cross_validation.KFold(n_instances, n_folds=5, shuffle=True)
for train_indices, test_indices in kfolds:
    train_examples = examples[train_indices]
    train_targets = targets[train_indices]

    nn.fit(train_examples, train_targets, n_epochs=1000, learning_rate=0.3)

    test_examples = examples[test_indices]
    test_targets = targets[test_indices]

    count = 0.0
    for example, target in zip(test_examples, test_targets):
        count += target[nn.predict(example)]
    print "accuracy:", count / len(test_targets)
