#
# Project 1, starter code part b
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt


NUM_FEATURES = 8

learning_rate = 10**-7
ridge_param = 10**-3
epochs = 500
batch_size = 32
num_neurons = 30
seed = 10
np.random.seed(seed)

#read and divide data into test and train sets 
cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
Y_data = (np.asmatrix(Y_data)).transpose()

idx = np.arange(X_data.shape[0])
np.random.shuffle(idx)
X_data, Y_data = X_data[idx], Y_data[idx]


m = 3* X_data.shape[0] // 10

# validX and validY are the last 70% of the dataset
validX, validY = X_data[m:], Y_data[m:]
validX = (validX- np.mean(validX, axis=0))/ np.std(validX, axis=0)

#experiment with small datasets
n = validX.shape[0]

#create validation set
testX, testY = X_data[m:], Y_data[m:]
testX = (testX- np.mean(testX, axis=0))/ np.std(testX, axis=0)

# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, [None, 1])

# Build the graph for the deep net
weights_h = tf.Variable(tf.truncated_normal([NUM_FEATURES,num_neurons], stddev=1.0/math.sqrt(float(NUM_FEATURES)))) 
biases_h = tf.Variable(tf.zeros([num_neurons]))
h = tf.nn.relu(tf.matmul(x, weights_h) + biases_h)

weights = tf.Variable(tf.truncated_normal([num_neurons, 1], stddev=1.0 / math.sqrt(num_neurons), dtype=tf.float32), name='weights')
biases = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='biases')
y = tf.matmul(h, weights) + biases

ridge_loss = tf.reduce_mean(tf.square(y_ - y))
regularization = tf.nn.l2_loss(weights) + tf.nn.l2_loss(weights_h)
loss = tf.reduce_mean(ridge_loss + ridge_param*regularization)

#Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

shuffle = np.arange(n)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	train_err = []
	for i in range(epochs):
		np.random.shuffle(shuffle)
		validX, validY = validX[shuffle], validY[shuffle]
		# implementing mini-batch GD
		for start in range(0, n-batch_size, batch_size):
			train_op.run(feed_dict={x: validX[start:start+batch_size], y_: validY[start:start+batch_size]})	
		err = loss.eval(feed_dict={x: validX, y_: validY})
		train_err.append(err)

		if i % 100 == 0:
			print('iter %d: validation error %g'%(i, train_err[i]))
	saver.save(sess, "/models/1b_1a_model.ckpt")


# plot learning curves
plt.figure(1)
plt.plot(range(epochs), train_err)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Validation Error')
plt.show()