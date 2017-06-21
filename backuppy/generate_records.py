import os
import tensorflow as tf 
import sys
import array
import numpy as np

FEATURE_SIZE = 500
NUM_CLASS = 2

def main(argv):
    cwd = os.getcwd()

    print one_hot(0)

    with tf.python_io.TFRecordWriter("train.tfrecords") as writer:
        for index in (0, NUM_CLASS-1):
            class_path = cwd + '/' + str(index) + "/"
            for file_name in os.listdir(class_path):
                content = getdata(class_path+file_name)
                input_index = one_hot(index)
                

                example = tf.train.Example()
                example.features.feature["signal"].float_list.value.extend(content)
                example.features.feature["label"].int64_list.value.extend(input_index)
                writer.write(example.SerializeToString())
 

    writer.close()
'''
    filename_queue = tf.train.string_input_producer(["train.tfrecords"])

    batchSize = 10
    min_after_dequeue = 8
    capacity = min_after_dequeue + 3 * batchSize
    num_threads = 2
    label, features = read_and_decode(filename_queue)
    batch_labels, batch_features = tf.train.shuffle_batch([label, features], batch_size= batchSize, num_threads= num_threads, capacity= capacity, min_after_dequeue = min_after_dequeue)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(3):
            val, l = sess.run([batch_features, batch_labels])
            print(val.shape, l[0][1])
'''
def one_hot(i):
    a = np.zeros(NUM_CLASS, dtype = int)
    a[i] = 1
    return a


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




def getdata(file_path):
    content = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            content.append(float(line))

    return content

if __name__ == '__main__':
    main(sys.argv)
