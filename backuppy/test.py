
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time

NUM_CLASS = 2
FEATURE_SIZE = 500

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


# Target log path
logs_path = '/tmp/tensorflow/logs'
writer = tf.summary.FileWriter(logs_path)

# Text file containing words for training
#training_file = 'signal.txt'
#neg_file = 'random.txt'

def read_data(fname):
    content = []
    with open(fname) as f:
        for l in f.readlines():
            content.append(float(l))
    return content

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
    features={
        "label": tf.FixedLenFeature([NUM_CLASS], tf.int64),
        "signal": tf.FixedLenFeature([FEATURE_SIZE], tf.float32),
    })

    label_out = features["label"]
    feature_out = features["signal"]

    return label_out, feature_out




#input 
filename_queue = tf.train.string_input_producer(["train.tfrecords"])

batchSize = 10
min_after_dequeue = 8
capacity = min_after_dequeue + 3 * batchSize
num_threads = 1
label, features = read_and_decode(filename_queue)
batch_labels, batch_features = tf.train.shuffle_batch([label, features], batch_size= batchSize, num_threads= num_threads, capacity= capacity, min_after_dequeue = min_after_dequeue)




# Parameters
learning_rate = 0.001
training_iters = 5000
display_step = 1000
n_input = 500

# number of units in RNN cell
n_hidden = 16
vocab_size = 2


# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, 2])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}

def RNN(x, weights, biases):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()


with tf.Session() as session:
    session.run(init)
    step = 0
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)

    try:
        while not coord.should_stop():
            x_r, y_r = session.run([batch_features, batch_labels])
            # Generate a minibatch. Add some randomness on selection process.
            _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: x_r, y: y_r})
            loss_total += loss
            acc_total += acc


            if (step+1) % display_step == 0:
                print("Iter= " + str(step+1) + ", Average Loss= " + \
                  "{:.6f}".format((loss_total/display_step)) + ", Average Accuracy= " + \
                  "{:.2f}%".format((100*acc_total/display_step)))
                acc_total = 0
                loss_total = 0
            step += 1

            if step > training_iters:
                break

    except tf.errors.OutOfRangeError:
        print ('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

	test_signal = read_data('testset/output_sun830.txt') 
       
        test_label = [0, 1]
        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: test_signal, y: test_label})

        print ("test: %f".format(acc))

        session.close()
        print("Optimization Finished!")
        print("Elapsed time: ", elapsed(time.time() - start_time))

