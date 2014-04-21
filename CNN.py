import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

class LeNetConvPoolLayer(object):
	def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
		assert image_shape[1] == filter_shape[1]
		self.input = input

		# initialize weight values. the fan_in of each hidden neuron
		# is restricted by the size of the receptive field
		fan_in = numpy.prod(filter_shape[1:])
		fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(poolsize))
		W_bound = numpy.sqrt(6./(fan_in+fan_out))
		self.W = theano.shared(numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX), borrow=True)

		# the bias is a 1D tensor. one bias per output feature map
		b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
		self.b = theano.shared(value=b_values, borrow=True)

		# convolve input feature maps with filters
		conv_out = conv.conv2d(input=input, filters=self.W, filter_shape=filter_shape, image_shape=image_shape)

		# downsample each feature map individually using maxpooling
		pooled_out = downsample.max_pool_2d(input=conv_out, ds=poolsize, ignore_border=True)

		# add bias to the term
		self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

		# parameters of this layer
		self.params = [self.W, self.b]
