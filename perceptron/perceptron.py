
import random

class Perceptron(object):
    def __init__(self, inputs, learning_rate):
        self.inputs = inputs
        self.weights = [random.uniform(-1, 1) for _ in self.inputs[0].inputs]
        self.learning_rate = learning_rate

    # activation function, return 1 if positive else -1 
    def sig(self, n):
        return 1 if n == abs(n) else -1

    # calculate the weighted sum all inputs of the given data
    def weighted_sum(self, input_data):
        res = 0
        for i, data in enumerate(input_data):
            val = data * self.weights[i]
            res += val
        return res

    # prediction of the perceptron
    def predict(self, input_data):
        # weighted sum = x1*w1 + x2*w2 + x3*w3 + ...
        weighted_sum = self.weighted_sum(input_data)
        prediction = self.sig(weighted_sum)
        return prediction

    def train(self, train_data, target):
        # if not train_data:
        #     train_data = self.inputs
        prediction = self.predict(train_data)
        error = prediction - target

        # adjusting the weights
        for i in range(len(self.weights)):
            self.weights[i] += error * train_data[i] * self.learning_rate

    def accuracy(self, predictions, labels):
        res = []
        for i, j in zip(predictions, labels):
            res.append(i == j)

        return (sum(res) / len(res)) * 100

