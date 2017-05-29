from deap import creator, base, tools, algorithms
import random
import numpy
import tensorflow as tf
from PIL import Image
from io import BytesIO

imageSideLen = 28
flatImageSize = imageSideLen ** 2


# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile("retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	_ = tf.import_graph_def(graph_def, name='')

# Make predictions
with tf.Session() as sess:

	# Feed the image_data as input to the graph and get first prediction
	softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')


# This function converts the individuals from the GA to .jpg file encodings used in TensorFlow classifiers. 
def individualToJpgFormat(individual):

	# Made a byte buffer, saved the PIL Image into it, got its value and passed it in.
	# Exact same as reading a .jpg file in using tf.gfile.FastGFile()
	newIndividual = [0 for element in individual]
	for index in range(len(individual)):
		element = individual[index]

		# Also color the ajacent pixels.  This should magnify the reward for proper pixel placements, and magnify the penalty for noise.
		if element == 1:
			a = index - imageSideLen - 1
			b = index - imageSideLen
			c = index - imageSideLen + 1

			d = index - 1
			e = index
			f = index + 1

			g = index + imageSideLen - 1
			h = index + imageSideLen
			i = index + imageSideLen + 1
			
			allIndicies = [a, b, c, d, e, f, g, h, i]
			for subIndex in allIndicies:
				if (subIndex >= 0) and (subIndex < flatImageSize):
					newIndividual[subIndex] = 255

	individual = numpy.asarray(newIndividual).astype(dtype=numpy.uint8).reshape(imageSideLen, imageSideLen)
	im = Image.fromarray(individual)
	imageBuf = BytesIO()
	im.save(imageBuf, format="JPEG")
	image_data = imageBuf.getvalue()

	return image_data


def getFitness(individual):

	# Get image data from individual
	image_data = individualToJpgFormat(individual)

	# Make a prediction
	predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

	# Return prediction
	return (predictions[0][0], )


#========DEAP GLOBAL VARIABLES (viewable by SCOOP)========

# Create Individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def assignPixelValues():
	perSpots = 0.01
	var = random.uniform(0, 1)
	if var <= perSpots:
		return 1
	else:
		return 0

# Create Toolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", assignPixelValues)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, flatImageSize)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Continue filling toolbox...
toolbox.register("evaluate", getFitness)
toolbox.register("mate", tools.cxUniform, indpb=0.7)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selRoulette)

#========


# Initialize variables to use eaSimple
numPop = 20
numGen = 250
pop = toolbox.population(n=numPop)
hof = tools.HallOfFame(numPop * numGen)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean)
stats.register("std", numpy.std)
stats.register("min", numpy.min)
stats.register("max", numpy.max)

# Launch genetic algorithm
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=numGen, stats=stats, halloffame=hof, verbose=True)

# Retrieve results and save
percentile = 0.001
numIndicies = 1 / percentile
indicies = sorted(list(set([int(round(value)) for value in numpy.linspace(0, len(hof) - 1, num=numIndicies)])))
indicies.reverse()
for i in range(len(indicies)):
	index = indicies[i]
	individual = hof[index]
	fitness = individual.fitness.values[0]
	jpgFormat = individualToJpgFormat(individual)

	ifile = open('images/' + str(i) + '_' + str(fitness) + '.jpg', 'wb')
	ifile.write(jpgFormat)
	ifile.close()













