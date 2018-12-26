class Bird(object):
    def __init__(self, name):
        self.name = name

class UltraBird(Bird):
    def __init__(self, name, type):
        super().__init__(name)
        self.type = type

ub1 = UltraBird("caspy","sekisei")
print(ub1.name, ub1.type)
