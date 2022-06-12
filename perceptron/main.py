
# ------------------------------- required libraries --------------------------------- #

from perceptron import Perceptron
from point import Point
import pygame

# -------------------------------- global variables ---------------------------------- #

width = 500
height = 500
display = pygame.display.set_mode((width + 1, height + 1))
pygame.display.set_caption("Perceptron")

# ------------------------------ data seperation util -------------------------------- #

def data_split(data: list[list[int]], ratio: int) -> list[list[int]]:

    # using the ratio, determines the amount of train and test data
    train_quantity = int(len(data) * ratio)
    train_data, test_data = data[:train_quantity], data[train_quantity:]
    return train_data, test_data

# --------------------------------- main function ------------------------------------ #

def main():
    run = True
    # learning rate of the perceptron
    learning_rate = 0.01

    # data is generated using the Point() class which generates
    # random points and also has a label depending in the points
    points = [Point(width, height) for _ in range(500)]

    # 75% train data, 25% test data
    train_ratio = 0.75
    # seperating the train and the test data
    train_data, test_data = data_split(points, train_ratio)

    # the perceptron object
    neuron = Perceptron(train_data, learning_rate)

    # training the perceptron with the training data
    for point in train_data:
        neuron.train(point.inputs, point.label)
        
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # testing the perceptron with the test data
        test_predictions = []
        test_labels = [p.label for p in test_data]
        for point in test_data:
            prediction = neuron.predict(point.inputs)
            test_predictions.append(prediction)

        # visualizing the data values and the predicted results
        # if the colors of the dots and their outer circle are different then
        # the predicted value is not equal to the true value, i.e., the prediction 
        # is incorrect. Else, the prediction is correct
        for data, prediction in zip(test_data, test_predictions):
            label_color = (50, 200, 50) if data.label == 1 else (200, 50, 50)
            prediction_color = (50, 200, 50) if prediction == 1 else (200, 50, 50)
            pygame.draw.circle(display, label_color, data.inputs, radius = 1)
            pygame.draw.circle(display, prediction_color, data.inputs, radius = 4, width = 1)

        pygame.display.flip()

    # display the accuracy of the perceptron
    accuracy = neuron.accuracy(test_predictions, test_labels)
    print(f"accuracy :: {accuracy}%")
        
# ------------------------------ invoking main func -------------------------------- #

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------- #
