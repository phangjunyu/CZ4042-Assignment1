#
# Project 1, starter code part b
#

import tensorflow as tf
import numpy as np
import pylab as plt
import pickle
import math

NUM_FEATURES = 8

learning_rate = 0.5*10**-8
ridge_param = 10**-3
epochs = 500
batch_size = 32
num_neurons = [20, 40, 60, 80, 100]
seed = 10
np.random.seed(seed)
num_folds = 5

#read and divide data into test and train sets 
cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
Y_data = (np.asmatrix(Y_data)).transpose()

idx = np.arange(X_data.shape[0])
np.random.shuffle(idx)
pickle.dump( idx, open( "1b_3a.p", "wb" ) )
X_data, Y_data = X_data[idx], Y_data[idx]

m = 3* X_data.shape[0] // 10
trainX, trainY = X_data[m:], Y_data[m:]

trainX = (trainX- np.mean(trainX, axis=0))/ np.std(trainX, axis=0)

# experiment with small datasets
n = trainX.shape[0]

# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, [None, 1])

overall_error = []
for num_neuron in num_neurons:
	# Build the graph for the deep net
	weights_h = tf.Variable(tf.truncated_normal([NUM_FEATURES,num_neuron], stddev=1.0/math.sqrt(float(NUM_FEATURES)))) 
	biases_h = tf.Variable(tf.zeros([num_neuron]))
	h = tf.nn.relu(tf.matmul(x, weights_h) + biases_h)

	weights = tf.Variable(tf.truncated_normal([num_neuron, 1], stddev=1.0 / np.sqrt(num_neuron), dtype=tf.float32), name='weights')
	biases = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='biases')
	y = tf.matmul(h, weights) + biases

	ridge_loss = tf.square(y_ - y)
	regularization = tf.nn.l2_loss(weights) + tf.nn.l2_loss(weights_h)
	loss = tf.reduce_mean(ridge_loss + ridge_param*regularization)

	#Create the gradient descent optimizer with the given learning rate.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train_op = optimizer.minimize(loss)

	nf = n//num_folds
	shuffle = np.arange(trainX.shape[0])
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for fold in range(num_folds):
			start, end = nf*fold, (fold+1)*nf
			validX = trainX[start:end]
			validY = trainY[start:end]
			trainX_ = np.append(trainX[:start],trainX[end:],axis=0)
			trainY_ = np.append(trainY[:start],trainY[end:],axis=0)
			shuffle = np.arange(trainX_.shape[0])
			valid_errs = []
			for i in range(epochs):
				np.random.shuffle(shuffle)
				trainX_, trainY_ = trainX_[shuffle], trainY_[shuffle]
				# implementing mini-batch GD
				for s in range(0, n-batch_size, batch_size):
					train_op.run(feed_dict={x: trainX_[s:s+batch_size], y_: trainY_[s:s+batch_size]})
				if i % 100 == 0:
					print('finished training iter %d'%i)
			valid_errs.append(loss.eval(feed_dict={x:validX, y_:validY}))
			print("%d fold validation error is: %g" %(fold, valid_errs[-1]))		
	print('mean error = %g'% np.mean(valid_errs))
	overall_error.append(np.mean(valid_errs))	
# plot learning curves
fig = plt.figure(1)
plt.xlabel('number of neurons in hidden layer')
plt.ylabel('Cross-Validation Error')
plt.plot(num_neurons, overall_error)
plt.show()