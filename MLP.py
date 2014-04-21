import numpy
import theano
import theano.tensor as T
from Logistic_Regression import *

class HiddenLayer(object):
	def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
		self.input = input

		if W is None:
			W_values = numpy.asarray(rng.uniform(
				low=-numpy.sqrt(6. / (n_in + n_out)),
				high=numpy.sqrt(6. / (n_in + n_out)),
				size=(n_in, n_out)), dtype=theano.config.floatX)
			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4
		W = theano.shared(value=W_values, name='W', borrow=True)
		self.W = W

		if b == None:
			b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)
		self.b = b

		linear_output = T.dot(input, self.W) + self.b
		self.output = (linear_output if activation is None
				else activation(linear_output))

		self.params = [self.W, self.b]

class MLP(object):
	def __init__(self, rng, input, n_in, n_hidden, n_out):
		self.hiddenLayer = HiddenLayer(rng=rng, input=input, n_in=n_in, n_out=n_hidden,
					activation=T.tanh)

		self.logRegressionLayer = LogisticRegression(input=self.hiddenLayer.output,
					n_in = n_hidden,
					n_out = n_out)

		self.L1 = abs(self.hiddenLayer.W).sum() \
			+ abs(self.logRegressionLayer.W).sum()
		self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
			+ (self.logRegressionLayer.W ** 2).sum()

		self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
		self.errors = self.logRegressionLayer.errors

		self.params = self.hiddenLayer.params + self.logRegressionLayer.params
