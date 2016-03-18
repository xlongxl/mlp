import numpy as np
import util

class MLP:

    def __init__(self, n_inp, n_hid, n_out, debug=False):
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out

        self.debug = debug

        self.activate = util.logistic
        self.activate_deriv = util.logistic_deriv

        self.cost = util.square_err
        self.cost_deriv = util.square_err_deriv

        self.w1_init = np.random.randn(self.n_inp, self.n_hid)
        self.w2_init = np.random.randn(self.n_hid, self.n_out)

    def forward_pass(self, example):
        # insert bias
        self.a1 = np.atleast_2d(np.insert(example, 0, 1))

        self.z2 = np.dot(self.a1, self.w1)
        self.a2 = self.activate(self.z2)

        self.z3 = np.dot(self.a2, self.w2)
        self.a3 = self.activate(self.z3)

        return self.a3.copy()

    def backward_pass(self, target, learning_rate):
        delta_3 = np.multiply(
            self.cost_deriv(self.a3, target, scale=0.5),
            self.activate_deriv(self.z3))
        delta_2 = np.dot(delta_3, self.w2.T) * self.activate_deriv(self.z2)

        self.w2 -= learning_rate * np.dot(self.a2.T, delta_3)
        self.w1 -= learning_rate * np.dot(self.a1.T, delta_2)

    def fit(self, examples, targets, n_epochs=3000, learning_rate=0.1):
        assert examples.shape[0] == targets.shape[0]
        assert examples.shape[1] + 1 == self.n_inp
        assert targets.shape[1] == self.n_out

        self.w1 = self.w1_init.copy()
        self.w2 = self.w2_init.copy()

        for i in range(n_epochs):
            cost = 0.0
            for (example, target) in zip(examples, targets):
                self.forward_pass(example)
                self.backward_pass(target, learning_rate)
                cost += self.cost(self.a3, target, scale=0.5)
            if self.debug and i % (n_epochs / 100) == 0:
                print "Epoch", i
                print cost

    def predict(self, example):
        return np.argmax(self.forward_pass(example))
