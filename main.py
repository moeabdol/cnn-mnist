import numpy
import theano
import theano.tensor as T
from Utils import *
from Logistic_Regression import *
from MLP import *
from CNN import *
import matplotlib.pyplot as plt
from pylab import *

learning_rate = 0.1
n_epochs = 200
dataset = 'mnist.pkl.gz'
batch_size = 500
nkerns = [20, 50]
ishape = (28, 28)

datasets = load_data(dataset)

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

rng = numpy.random.RandomState(23455)

# allocate symbolic variables for the data
index = T.lscalar()	# index to the minibatch
x = T.matrix('x')	# rasterized images
y = T.ivector('y')	# labels are 1D vectors of int

###############
# build model #
###############
print '... building the model'

# reshape matrix of rasterized images of shape (batch_size, 28, 28)
# to a 4D tensor that is compatible with  out LeNetConvPoolLayer
layer0_input = x.reshape((batch_size, 1, 28, 28))

# construct the first convolutional pooling layer
layer0 = LeNetConvPoolLayer(rng, input=layer0_input, image_shape=(batch_size, 1, 28, 28),
			filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))

# construct the second convolutional pooling layer
layer1 = LeNetConvPoolLayer(rng, input=layer0.output, image_shape=(batch_size, nkerns[0], 12, 12),
			filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))

# the hidden layer being fully connected, it operates on 2D matrices
# of shape (batch_size, num_pixels)
layer2_input = layer1.output.flatten(2)

# construct a fully connected sigmoidal layer
layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1]*4*4, n_out=500, activation=T.tanh)

# classify the values of the fully connected sigmoidal layer
layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

# the cost to minimize during training
cost = layer3.negative_log_likelihood(y)

test_model = theano.function(inputs=[index],
		outputs=layer3.errors(y),
		givens={x: test_set_x[index * batch_size: (index + 1) * batch_size],
			y: test_set_y[index * batch_size: (index + 1) * batch_size]})

valid_model = theano.function(inputs=[index],
		outputs=layer3.errors(y),
		givens={x: valid_set_x[index * batch_size: (index + 1) * batch_size],
			y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

# model to show misclassified examples
misclassified_model = theano.function(inputs=[x], outputs=layer3.y_pred)

# create a list of all model parameters to be fit by
# gradient descent
params = layer3.params + layer2.params + layer1.params + layer0.params

# create a list of gradients for all model parameters
grads = T.grad(cost, params)

# specify how to update the parameters of the model
updates = []
for param_i, grad_i in zip(params, grads):
	updates.append((param_i, param_i - learning_rate * grad_i))

train_model = theano.function(inputs=[index],
		outputs=cost,
		updates=updates,
		givens={x: train_set_x[index * batch_size: (index + 1) * batch_size],
			y: train_set_y[index * batch_size: (index + 1) * batch_size]})

def train():
	print '... training the model'
	epoch = 0
	while(epoch < n_epochs):
		epoch = epoch + 1
		for minibatch_index in xrange(n_train_batches):
			train_model(minibatch_index)
		print 'training epoch %i' %epoch

def test():
	for minibatch_index in xrange(n_test_batches):
		minibatch_losses = [test_model(minibatch_index)]
		test_score = numpy.mean(minibatch_losses)
		print 'testing minibatch %i with mean error = %f' %(minibatch_index + 1, test_score)

def show_misclassified():
	for i in xrange(n_test_batches * batch_size):
		image = test_set_x[i].eval()
		label = test_set_y[i].eval()

		# a smart workaround to construct a fake (500, 784) matrix of 
		# the first row as a vector of the image we want to classify
		# and the rest of the matrix filled with zeros
		image_matrix = numpy.zeros((500, 784), dtype=theano.config.floatX)
		for a in xrange(784):
			image_matrix[0, a] = image[a]

		prediction = misclassified_model(image_matrix)[0]
		if prediction != label:
			print 'misclassified example found in test set at index %i' %i
			# show actual image
			image = image.reshape(28, 28)
			plt.imshow(image, cmap=cm.gray)
			plt.xlabel('Actual ' + str(label))
			plt.ylabel('Prediction ' + str(prediction))
			plt.show()














