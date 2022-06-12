
import random

class Perceptron(object):
    def __init__(self, inputs: list[list[int]], learning_rate: float):
        self.inputs = inputs
        self.weights = [random.uniform(-1, 1) for _ in self.inputs[0].inputs]
        self.learning_rate = learning_rate

    # activation function, return 1 if positive else -1 
    def sig(self, n) -> int:
        return 1 if n > 0 else -1

    # calculate the weighted sum all inputs of the given data
    def weighted_sum(self, input_data: list[int]) -> int:
        res = 0
        for data, weight in zip(input_data, self.weights):
            val = data * weight
            res += val
        return res
 
    # prediction of the perceptron
    def predict(self, input_data: list[int]) -> int:
        # weighted sum = x1*w1 + x2*w2 + x3*w3 + ...
        weighted_sum = self.weighted_sum(input_data)
        prediction = self.sig(weighted_sum)
        return prediction

    # training the perceptron with known data
    def train(self, train_data: list[int], target: int) -> None:
        prediction = self.predict(train_data)
        error = target - prediction

        # gradient descent
        # nudging the weights by calculating delta weight
        for i in range(len(self.weights)):
            delta_weight = error * train_data[i] * self.learning_rate
            self.weights[i] += delta_weight

    # to calculate the accuracy of the perceptron
    def accuracy(self, predictions: list[int], labels: list[int]) -> float:
        res = []
        for prediction, label in zip(predictions, labels):
            res.append(prediction == label)

        accuracy_percentage = (sum(res) / len(res)) * 100
        return accuracy_percentage
