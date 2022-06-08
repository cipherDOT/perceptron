import random

class Point(object):
    def __init__(self):

        # make two random points
        self.x = int(random.uniform(0, 500))
        self.y = int(random.uniform(0, 500))

        # check if x > y and set the label acc. to that.
        self.label = 1 if self.x > self.y else -1

        # utilities to use the data
        self.inputs = (self.x, self.y)
