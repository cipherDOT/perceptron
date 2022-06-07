import random

class Point(object):
    def __init__(self):
        self.x = int(random.uniform(0, 1) * 10)
        self.y = int(random.uniform(0, 1) * 10)
        self.label = 1 if self.x > self.y else -1
        self.data = (self.x, self.y, self.label)
        self.inputs = (self.x, self.y)
