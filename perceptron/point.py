import random

class Point(object):
    def __init__(self, upper_limit_x: int, upper_limit_y: int) -> None:

        # make two random points
        self.x = random.randint(0, upper_limit_x)
        self.y = random.randint(0, upper_limit_y)

        # check if x > y and set the label acc. to that.
        self.label = 1 if self.x > self.y else -1

        # utilities to use the data
        self.inputs = (self.x, self.y)
