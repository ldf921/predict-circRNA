from reader import process_feature
import random 
import numpy as np
import tensorflow as tf
from model import RNNModel

def get_feature_label(data):
	length = np.array(list(map(lambda x : x.length, data)), dtype = np.int32)
	max_length = np.max(length) 
	feature_dim = data[0].feature_dim
	batch_size = len(data)
	x = np.zeros([batch_size, max_length, feature_dim])
	y = np.zeros([batch_size])

	for i in range(len(data)):
		x[i, : data[i].length, :] = data[i].features
		y[i] = data[i].label

	return x, y, length

model = RNNModel()
sess = tf.Session()

data = process_feature()

steps = 100
batch_size = 20
learning_rate = 0.01

for t in range(0, steps):
	x, y, length = get_feature_label(random.sample(data, batch_size))
	loss = model.train(sess, x, y, length, learning_rate)
	print('step = %d, loss = %.5f' % (t, loss))
