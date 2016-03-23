import numpy as np
import util

class MLP:

    def __init__(self, layer_sizes, debug=False):
        assert len(layer_sizes) >= 2
        self.layer_sizes = layer_sizes

        self.debug = debug

        self.activate = util.logistic
        self.activate_deriv = util.logistic_deriv

        self.cost = util.square_err
        self.cost_deriv = util.square_err_deriv

        self.W_init = []
        for i in range(1, len(layer_sizes)):
            weight = np.random.randn(layer_sizes[i-1], layer_sizes[i])
            self.W_init.append(weight)

    def forward_pass(self, example):
        self.A = [None] * len(self.layer_sizes)
        self.Z = [None] * len(self.layer_sizes)

        # insert bias
        self.A[0] = np.atleast_2d(np.insert(example, 0, 1))

        for i in range(1, len(self.layer_sizes)):
            self.Z[i] = np.dot(self.A[i-1], self.W[i-1])
            self.A[i] = self.activate(self.Z[i])

        return self.A[-1].copy()

    def backward_pass(self, target, learning_rate):
        Deltas = [None] * len(self.layer_sizes)

        Deltas[-1] = np.multiply(
            self.cost_deriv(self.A[-1], target, scale=0.5),
            self.activate_deriv(self.Z[-1]))

        for i in range(len(self.layer_sizes) - 2, 0, -1):
            Deltas[i] = np.multiply(
                np.dot(Deltas[i+1], self.W[i].T),
                self.activate_deriv(self.Z[i]))

        for i in range(len(self.layer_sizes) - 1):
            self.W[i] -= learning_rate * np.dot(self.A[i].T, Deltas[i+1])

    def fit(self, examples, targets, n_epochs=3000, learning_rate=0.1):
        assert examples.shape[0] == targets.shape[0]
        assert examples.shape[1] + 1 == self.layer_sizes[0]
        assert targets.shape[1] == self.layer_sizes[-1]

        self.W = []
        for weight in self.W_init:
            self.W.append(weight.copy())

        for i in range(n_epochs):
            cost = 0.0
            for (example, target) in zip(examples, targets):
                self.forward_pass(example)
                self.backward_pass(target, learning_rate)
                cost += self.cost(self.A[-1], target, scale=0.5)
            if self.debug and i % (n_epochs / 100) == 0:
                print "Epoch", i
                print cost

    def predict(self, example):
        return np.argmax(self.forward_pass(example))
