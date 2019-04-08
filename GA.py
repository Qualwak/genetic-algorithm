import numpy
from PIL import ImageDraw


def select(pop, fitness, num_parents):
    parents = numpy.empty((num_parents, pop.shape[1], pop.shape[2]))
    for parent_num in range(num_parents):
        index_most_fitness = (numpy.abs(numpy.asarray(fitness) - 0)).argmin()
        parents[parent_num, :, :] = pop[index_most_fitness, :, :]
        fitness[index_most_fitness] = -99999999999
    return parents


def crossover_function(parents, offspring_size, input_image, img_new, use_inverse_img):
    offspring = numpy.empty(offspring_size)

    image = ImageDraw.Draw(img_new)

    width, height = 5, 5
    for chr in range(offspring_size[0]):
        first = chr % parents.shape[0]
        second = (chr + 1) % parents.shape[0]
        if use_inverse_img:
            for i in range(input_image.shape[0]):
                x, y = i % 512, i // 512

                first_diff, second_diff = 0, 0
                for j in range(3):
                    first_diff += 255 - abs(input_image[i][j] - parents[first][i][j])
                    second_diff += 255 - abs(input_image[i][j] - parents[second][i][j])
                if first_diff < second_diff and first_diff < 20:
                    image.rectangle((x - width, y - height, x + width, y + height),
                                    fill=(int(parents[first][i][0]), int(parents[first][i][1]),
                                          int(parents[first][i][2])))
                elif second_diff < 20:
                    image.rectangle((x - height, y - width, x + height, y + width),
                                    fill=(int(parents[second][i][0]), int(parents[second][i][1]),
                                          int(parents[second][i][2])))
            offspring[chr] = numpy.array(img_new.getdata())
        else:
            for i in range(input_image.shape[0]):
                x, y = i % 512, i // 512
                first_diff, second_diff = 0, 0
                for j in range(3):
                    first_diff += abs(input_image[i][j] - parents[first][i][j])
                    second_diff += abs(input_image[i][j] - parents[second][i][j])
                if second_diff < 15:
                    image.rectangle((x - height, y - width, x + height, y + width), fill=(
                        int(parents[second][i][0]), int(parents[second][i][1]),
                        int(parents[second][i][2])))
                elif first_diff < 15 and first_diff < second_diff:
                    image.rectangle((x - width, y - height, x + width, y + height),
                                    fill=(int(parents[first][i][0]), int(parents[first][i][1]),
                                          int(parents[first][i][2])))
            offspring[chr] = numpy.array(img_new.getdata())
    return offspring


def mutation(crossover_list):
    for index in range(crossover_list.shape[0]):
        gen_number = numpy.random.randint(0, crossover_list.shape[1] // 5)
        for i in range(gen_number):
            crossover_list[index, numpy.random.randint(0, crossover_list.shape[1])] = numpy.random.uniform(-255.0, 256.0, 3)
    return crossover_list


def fitness_function(img_array, population, use_inverse_img):
    fitness = list()
    if use_inverse_img:
        for chromosome in population:
            difference = 0
            for k in range(img_array.size // 3):
                if 255 - abs(img_array[k][0] - chromosome[k][0]) >= 100:
                    difference += 2000
                else:
                    difference += 255 - abs(img_array[k][0] - chromosome[k][0])
                if 255 - abs(img_array[k][1] - chromosome[k][1]) >= 100:
                    difference += 2000
                else:
                    difference += 255 - abs(img_array[k][1] - chromosome[k][1])
                if 255 - abs(img_array[k][2] - chromosome[k][2]) >= 100:
                    difference += 2000
                else:
                    difference += 255 - abs(img_array[k][2] - chromosome[k][2])
            fitness.append(difference)
    else:
        for chromosome in population:
            difference = 0
            for k in range(img_array.size // 3):
                if img_array[k][0] - chromosome[k][0] >= 100:
                    difference += 2000
                else:
                    difference += 255 - img_array[k][0] - chromosome[k][0]
                if img_array[k][1] - chromosome[k][1] >= 100:
                    difference += 2000
                else:
                    difference += 255 - img_array[k][1] - chromosome[k][1]
                if img_array[k][2] - chromosome[k][2] >= 100:
                    difference += 2000
                else:
                    difference += 255 - img_array[k][2] - chromosome[k][2]
            fitness.append(difference)
    return fitness
