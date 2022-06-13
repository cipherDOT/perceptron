import random

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

    @staticmethod
    def line_function(x: int, return_point = False) -> tuple[int] | int:
        # f(x) = mx + c
        y = x

        if return_point:
            return (x, y)
        
        return y

