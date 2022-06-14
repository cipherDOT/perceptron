# perceptron
A perceptron model built using python 3.10

## The idea behind a Perceptron:
Perceptron is a algorithm designed with the idea of mimicking the working and function of a neuron in a human brain. A single neuron(Perceptron) can be used for binary classifications only ([reference](!https://en.wikipedia.org/wiki/Perceptron)). This algorithm uses the concept of gradient descent to minimize the error of prediction.

### Perceptron model (code):

```
class Perceptron(object):
    def __init__(self, inputs: list[list[int]], learning_rate: float):
        self.inputs = inputs
        self.weights = [random.uniform(-1, 1) for _ in self.inputs[0].inputs]
        self.learning_rate = learning_rate

```

The weight values are randomly set, according to the number of inputs in the data. The range of the weights is between -1 to 1 (excluding -1 and 1). 

The weighted sum is calculated by, 

```
def weighted_sum(self, input_data: list[int]) -> int:
    return sum([data * weight for data, weight in zip(input_data, self.weights)])

```
 and, the activation function is given by the sign by,
 
 ```
def sig(self, n: int) -> int:
        return 1 if n > 0 else -1
 ```
 which return the sign of the given integer, n.
 The prediction is done by calculating the weighted sum of the inputs and passing it through the activation function.
 
 ```
 def predict(self, input_data: list[int]) -> int:
        weighted_sum = self.weighted_sum(input_data)
        prediction = self.sig(weighted_sum)
        return prediction
 ```
 
 Finally, the accuracy of the model is found by,
 
 ```
 def accuracy(self, predictions: list[int], labels: list[int]) -> float:
        res = []
        for prediction, label in zip(predictions, labels):
            res.append(prediction == label)

        accuracy_percentage = (sum(res) / len(res)) * 100
        return accuracy_percentage
```
In this project, the perceptron is trained to figure out, whether the given point lies above(label: -1) or below(label: 1) the line *y = x*.
Tho generate the data required for this process we use the data class "Point",

```
class Point(object):
    def __init__(self, x_upper: int, y_upper: int) -> None:

        # make two random points
        self.x = random.randint(0, x_upper)
        self.y = random.randint(0, y_upper)
        self.bias = 1

        # check if x < y and set the label acc. to that.
        self.label = 1 if self.line_function(self.x) < self.y else -1

        # utilities to use the data
        self.inputs = (self.x, self.y, self.bias)
        self.points = (self.x, self.y)
```
Using this class, we can generate points by,
```
points = [Point(width, height) for _ in range(1000)]
```
To split the training and the testing data, we use the utility function called data_split() that takes in a data and a ratio and splits the data according to the 
ratio, by means of the following,

```
def data_split(data, ratio):

    # using the ratio, determines the amount of train and test data
    train_quantity = int(len(data) * ratio)
    train_data, test_data = data[:train_quantity], data[train_quantity:]
    return train_data, test_data
```
To inititate the algorithm, i.e., to create the perceptron and train the perceptron with all the points in the training data,
```
neuron = Perceptron(train_data, learning_rate)
```

```
for point in train_data:
        neuron.train(point.inputs, point.label)
```
Next, the perceptron predicts the label of all the points in the test data, and the predictions are stored in a list called test_predictions

```
test_predictions = []
test_labels = [point.label for point in test_data]
for point in test_data:
    prediction = neuron.predict(point.inputs)
    test_predictions.append(prediction)
```
Finally, the accuracy of the perceptron is calculated by,
```
accuracy = neuron.accuracy(test_predictions, test_labels)
print(f"accuracy :: {accuracy}%")
```

## References:
Wikipedia article : https://en.wikipedia.org/wiki/Perceptron

Javapoint article : https://www.javatpoint.com/perceptron-in-machine-learning

 
 
 
