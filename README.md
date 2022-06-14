# # perceptron
# A perceptron model built using python 3.10

# ## The idea behind a Perceptron:
# Perceptron is a algorithm designed with the idea of mimicking the working and function of a neuron in a human brain. A single neuron(Perceptron) can be used for binary classifications only ([reference](!https://en.wikipedia.org/wiki/Perceptron)). This algorithm uses the concept of gradient descent to minimize the error of prediction.

# ### Perceptron model (code):

# ```
# class Perceptron(object):
#     def __init__(self, inputs: list[list[int]], learning_rate: float):
#         self.inputs = inputs
#         self.weights = [random.uniform(-1, 1) for _ in self.inputs[0].inputs]
#         self.learning_rate = learning_rate

# ```

# The weight values are randomly set, according to the number of inputs in the data. The range of the weights is between -1 to 1 (excluding -1 and 1). 

# The weighted sum is calculated by, 

# ```
# def weighted_sum(self, input_data: list[int]) -> int:
#     return sum([data * weight for data, weight in zip(input_data, self.weights)])

# ```
#  and, the activation function is given by the sign by,
 
#  ```
# def sig(self, n: int) -> int:
#         return 1 if n > 0 else -1
#  ```
#  which return the sign of the given integer, n.
#  The prediction is done by calculating the weighted sum of the inputs and passing it through the activation function.
 

 
 
