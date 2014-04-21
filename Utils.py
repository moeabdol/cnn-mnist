import cPickle
import gzip
import os
import theano
import theano.tensor as T
import numpy

def load_data(dataset):
	data_dir, data_file = os.path.split(dataset)
	if (not os.path.exists(dataset)) or (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
		import urllib
		origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
		print '... downloading data from %s' %origin
		urllib.urlretrieve(origin, dataset)

	# load the dataset
	print '... loading data'
	f = gzip.open(dataset, 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()

	train_set_x, train_set_y = shared_dataset(train_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)
	test_set_x, test_set_y = shared_dataset(test_set)

	sets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (train_set_x, train_set_y)]
	return sets

def shared_dataset(data_xy, borrow=True):
	data_x, data_y = data_xy
	shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
	shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
	return shared_x, T.cast(shared_y, 'int32')

	
