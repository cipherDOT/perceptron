from perceptron import Perceptron
from point import Point

if __name__ == '__main__':
    learning_rate = 0.1
    points = [Point() for _ in range(100)]
    ratio = int(len(points) * 0.75)
    train_data, test_data = points[:ratio], points[ratio:]
    # neuron = Perceptron(train_data, learning_rate)

    for _ in range(100):
        neuron = Perceptron(train_data, learning_rate)
        for _ in range(5):
            for point in train_data:
                neuron.train(point.inputs, point.label)

        test_predictions = []
        for point in test_data:
            prediction = neuron.predict(point.inputs)
            test_predictions.append(prediction)
            

        accuracy = neuron.accuracy(test_predictions, [p.label for p in test_data])
        print(f"accuracy :: {accuracy}%")

        if accuracy >= 95:
            with open('note.txt', 'a') as f:
                data = f"{neuron.weights} : {accuracy}\n"
                f.write(data)
                f.close()
