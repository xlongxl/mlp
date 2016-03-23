import sys
sys.path.append('../lib')

import numpy as np
import mlp

from sklearn import cross_validation

# http://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame
fname = 'data/tic_tac_toe.csv'

n_instances = 958

n_attributes = 9
attributes_cols = range(9)

n_classes = 2
classes_col = 9

attribute_dict = {'o': 0.0, 'b': 0.5, 'x': 1.0}
examples = np.genfromtxt(
    fname, delimiter=',', usecols=attributes_cols,
    converters=dict.fromkeys(attributes_cols, lambda x: attribute_dict[x]))

class_dict = {'negative': 0, 'positive': 1}
classes = np.genfromtxt(
    fname, delimiter=',', usecols=classes_col,
    converters={classes_col: lambda x: class_dict[x]})

targets = np.zeros((n_instances, n_classes))
for class_, target in zip(classes, targets):
    target[class_] = 1.0

nn = mlp.MLP([n_attributes + 1, 6, 6, n_classes], debug=True)

kfolds = cross_validation.KFold(n_instances, n_folds=5, shuffle=True)
for train_indices, test_indices in kfolds:
    train_examples = examples[train_indices]
    train_targets = targets[train_indices]

    nn.fit(train_examples, train_targets, n_epochs=1000, learning_rate=0.05)

    test_examples = examples[test_indices]
    test_targets = targets[test_indices]

    count = 0.0
    for example, target in zip(test_examples, test_targets):
        count += target[nn.predict(example)]
    print "accuracy:", count / len(test_targets)
