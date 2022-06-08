from perceptron import Perceptron
from point import Point

if __name__ == '__main__':
    # learning rate of the perceptron
    learning_rate = 0.1

    # data is generated using the Point() class which generates
    # random points and also has a label depending in the points
    points = [Point() for _ in range(100)]

    # 75% train data, 25% test data
    train_ratio = 0.75
    train_quantity = int(len(points) * train_ratio)

    # seperating the train and the test data
    train_data, test_data = points[:train_quantity], points[train_quantity:]

    # the perceptron object
    neuron = Perceptron(train_data, learning_rate)

    # training the perceptron with the training data
    for point in train_data:
        neuron.train(point.inputs, point.label)

    # testing the perceptron with the test data
    test_predictions = []
    test_labels = [p.label for p in test_data]
    for point in test_data:
        prediction = neuron.predict(point.inputs)
        test_predictions.append(prediction)
        

    # display the accuracy of the perceptron
    accuracy = neuron.accuracy(test_predictions, test_labels)
    print(f"accuracy :: {accuracy}%")

