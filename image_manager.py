import numpy
from PIL import Image


class ImageManager:

    def __init__(self):
        self.format = '.jpg'
        self.use_inverse_img = False
        self.step_destination = 'output/generation'
        self.final_destination = 'output/final_output'
        self.as_input = Image.open("input_imgs/dog" + self.format)
        self.width, self.height = self.as_input.size
        self.weights = self.width * self.height

    def create(self, name, index, list):
        new = Image.new('RGB', (self.width, self.height))
        new.putdata([(i[0], i[1], i[2]) for i in list[index]])
        new.save(name + self.format)

    def get_array(self):
        return numpy.array(self.as_input.getdata())

    @staticmethod
    def choose_best(fitness):
        return numpy.where(numpy.max(fitness) == fitness)[0][0]
