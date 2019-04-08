import numpy
import GA
from PIL import Image
from image_manager import ImageManager

image = ImageManager()


def cycle(size_of_population, total_population, mating_parents, colors, img_new, gen):

    fitness = GA.fitness_function(image.get_array(), total_population, image.use_inverse_img)
    parents = GA.select(total_population, fitness, mating_parents)

    offspring_crossover = GA.crossover_function(parents, (size_of_population[0] - parents.shape[0], image.weights
                                                     , colors), image.get_array(), img_new, image.use_inverse_img)

    offspring_mutation = GA.mutation(offspring_crossover)

    total_population[0:parents.shape[0], :, :] = parents
    total_population[parents.shape[0]:, :, :] = offspring_mutation

    image.create(image.step_destination + str(gen), 0, total_population)


def main(gens, number_of_solutions, mating_parents, colors):
    img_new = Image.new('RGB', (image.width, image.height), 'WHITE')

    size_of_population = (number_of_solutions, image.weights, colors)

    total_population = numpy.random.randint(low=0, high=256, size=size_of_population)

    for gen in range(gens):
        print("Number of generation: {}".format(gen))

        cycle(size_of_population, total_population, mating_parents, colors, img_new, gen)

    fitness = GA.fitness_function(image.get_array(), total_population, image.use_inverse_img)

    image.create(image.final_destination, image.choose_best(fitness), total_population)


if __name__ == "__main__":
    use_inverse_img = True
    main(17, 20, 10, 3)
