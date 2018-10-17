#
# Project 1, Question 1A, Part 4B
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt


# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

NUM_FEATURES = 36
NUM_CLASSES = 6

learning_rate = 0.01
epochs = 1000
batch_size = 32 
num_neuron = 15
seed = 10
np.random.seed(seed)
ridge_params = [0, 10**-3, 10**-6, 10**-9, 10**-12]

#read train data
train_input = np.loadtxt('sat_train.txt',delimiter=' ')
trainX, train_Y = train_input[:,:36], train_input[:,-1].astype(int)
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))
train_Y[train_Y == 7] = 6

trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix

#read test data
test_input = np.loadtxt('sat_test.txt',delimiter=' ')
testX, test_Y = test_input[:,:36], test_input[:,-1].astype(int)
testX = scale(testX, np.min(testX, axis=0), np.max(testX, axis=0))
test_Y[test_Y == 7] = 6

testY = np.zeros((test_Y.shape[0], NUM_CLASSES))
testY[np.arange(test_Y.shape[0]), test_Y-1] = 1 #one hot matrix


# experiment with small datasets
# trainX = trainX[:1000]
# trainY = trainY[:1000]

n = trainX.shape[0]

# testX = testX[:1000]
# testY = testY[:1000]


# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])


test_accs = []
# Build the graph for the deep net
for ridge_param in ridge_params:
    weights_h = tf.Variable(tf.truncated_normal([NUM_FEATURES,num_neuron], stddev=0.001)) 
    biases_h = tf.Variable(tf.zeros([num_neuron]))

    weights = tf.Variable(tf.truncated_normal([num_neuron, NUM_CLASSES], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='weights')
    biases  = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')

    h = tf.nn.relu(tf.matmul(x, weights_h) + biases_h)
    logits = tf.matmul(h, weights) + biases

    ridge_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    regularization = tf.nn.l2_loss(weights) + tf.nn.l2_loss(weights_h)
    loss = tf.reduce_mean(ridge_loss + ridge_param*regularization)

    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    train_acc = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            for start in range(0, n-batch_size, batch_size):
                train_op.run(feed_dict={x: trainX[start:start+batch_size], y_: trainY[start:start+batch_size]})
            if i %100 == 0:
                print("iter %d: training accuracy %g"%(i,accuracy.eval(feed_dict={x: trainX, y_: trainY})))
        test_accs.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
        print('test accuracy %g'%(test_accs[-1]))
        sess.close()

# plot learning curves
plt.figure(1)
plt.plot(np.log10(ridge_params), test_accs)
plt.xlabel('Exponent of Decay Parameter')
plt.ylabel('Test Accuracy')
plt.show()

