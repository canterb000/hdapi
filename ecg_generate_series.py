#!/usr/bin/env python

import os
import sys
from matplotlib import pyplot as pl
import numpy as np
from scipy import signal
import tensorflow as tf
import array


#training records saving

#dataset parameters
NUM_CLASS = 12

#parsing .dat file parameters   + downsampling parameters
FILE_START_POSITION = 0x577a  #in byte
FILE_LEN_POSITION = 0x5776    #in byte
BIT_DEPTH = 16
REF_VOLTAGE = 0.3
LEADS_NUM = 13
FREQUENCE = 1024
DATA_LEN_SECOND = 6
DATA_TYPE = np.uint16
DATA_SIZE_BYTE = 16/8
PLOT_POINT_START = 0
PLOT_POINT_STOP = None   #end of datafile

SUBSAMPLE_RATIO = 5
OUTPUT_CHANNEL = 1

#generate subsampled signal series parameters
SINGLE_FILE_LEN = 500
SUBSAMPLE_SIGNAL_GAP = 100
FEATURE_SIZE = 500

debug_current_dat_dir = ""
debug_current_dat_name = ""
debug_current_dat_currentpoint = 0 
writer = None

def main(argv):
    global writer
    global debug_current_dat_name
    with tf.python_io.TFRecordWriter("train.tfrecords") as tfwriter:
	writer = tfwriter
	current_dir = os.getcwd();
	print "current_dir: " + current_dir
	for parents, dirnames, filnames in os.walk(current_dir):
		for singledir in dirnames:
		    if singledir == "data001":
			print (str(singledir))
			for index in range(0, NUM_CLASS):
			    print index
			    data_subset_dir = current_dir + "/" + singledir + "/" + str(index) + "/"
			    print (str(data_subset_dir))
			    for subparents, subdir, subfilenames in os.walk(data_subset_dir):
				for subfilename in subfilenames:
				     sub_file_name, sub_file_extension = os.path.splitext(subfilename)
				     if sub_file_extension == ".dat": 
                                         # DEBUG: show all the files
				         print "generate a record for sub : " +str(subfilename) + "--dir: " + str(data_subset_dir)
				         debug_current_dat_name = sub_file_name
				         generate_single_record(subfilename, data_subset_dir, index)
    tfwriter.close()

def generate_single_record(filename, filedir, index):
    global debug_current_dat_dir


    filepath = filedir + filename
    print str(filepath)
    print ".../n"
    debug_current_dat_dir = filedir
    debug_current_dat_name = filename
    print "***********: " + debug_current_dat_dir + debug_current_dat_name
    parse_single_file(filepath, index)

def parse_single_file(filename, file_class):
    data_buffer = np.fromfile(filename, dtype = np.uint8)
    data_len = np.frombuffer(data_buffer, dtype = np.uint32, count = 1, offset = FILE_LEN_POSITION)
    data_len = data_len[0]
    data = np.int32(np.frombuffer(data_buffer, dtype=DATA_TYPE, count=data_len / DATA_SIZE_BYTE, offset = FILE_START_POSITION))
    data_scaled = 1000 * (data - (2**BIT_DEPTH / 2 )) * REF_VOLTAGE / (2**BIT_DEPTH)
    data_array = data_scaled.reshape((-1, LEADS_NUM))

    start_point = PLOT_POINT_START
    if PLOT_POINT_STOP is None:
        stop_point = data_len / DATA_SIZE_BYTE / LEADS_NUM / FREQUENCE * 1000
    else:
        stop_point = PLOT_POINT_STOP

    t = np.linspace(0, (stop_point - start_point) / 1024.*1000, stop_point - start_point)


#subsampling
    output_array = data_array[start_point : stop_point, OUTPUT_CHANNEL]
    resampled_array = signal.resample(output_array, len(output_array) / SUBSAMPLE_RATIO)
    print "origin length: " + str(len(output_array)) + " resampled array length: " + str(len(resampled_array))
    print "filename: " + str(filename) 
    generate_resampled_records(resampled_array, filename, file_class)


def generate_resampled_records(signal, name, signal_label):
    global debug_current_dat_currentpoint


    end_point = len(signal)
    current = 20
    while current < (end_point - SINGLE_FILE_LEN):
        debug_current_dat_currentpoint = current
        subsampled_signal = []
        for data in signal[current : current + SINGLE_FILE_LEN]:
            subsampled_signal.append(float(data))
        print("resample signal: from :" + str(name) + "label: " + str(signal_label) + " ---start----" )
        save_single_record(subsampled_signal, name, signal_label)
        save_to_txt(subsampled_signal, name, signal_label) 
        current += SUBSAMPLE_SIGNAL_GAP


def save_to_txt(subsampled_signal, name, signal_label):
    global debug_current_dat_dir
    global debug_current_dat_name
    global debug_current_dat_currentpoint


    output_name = debug_current_dat_dir + debug_current_dat_name + str(debug_current_dat_currentpoint) + ".txt"
    output = open(output_name, 'w')

    output_str = "".join(str(subsampled_signal))
    output.write(output_str)
    output.write(str(signal_label))
    output.close()

#write to recordfile
def save_single_record(subsampled_signal, name, signal_label):
	global writer
        content = subsampled_signal
        input_index = one_hot(signal_label)

        example = tf.train.Example()
        example.features.feature["signal"].float_list.value.extend(content)
        example.features.feature["label"].int64_list.value.extend(input_index)
#        example.features.feature["name"].float_list.value.extend(content)
        writer.write(example.SerializeToString())

def one_hot(index):
    temp = np.zeros(NUM_CLASS, dtype = int)
    temp[index] = 1
    return temp



if __name__ == '__main__':
    main(sys.argv)

