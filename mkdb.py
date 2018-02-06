#!/usr/bin/env python

import os
import tensorflow as tf 
import sys
import array
import numpy as np
from PIL import Image 
import cPickle as p
import shutil


NUM_CLASS = 2
DATASET_FOLDER = "ecgtestdatabase"
OUTPUT_BATCH_PATH = "testbatch_1"
length = width = 32 #32
picsize = length * width

def main(argv):
    global datalist
    global labellist
    global filenamelist

    datalist = []
    labellist = []
    filenamelist = []
    data = {}

    current_dir = os.getcwd()
    print "current_dir: " + current_dir
    for parents, dirnames, filnames in os.walk(current_dir):
        for singledir in dirnames:
            if singledir == DATASET_FOLDER:
                print (str(singledir))
                for index in range(0, NUM_CLASS):
                    print index
                    data_subset_dir = current_dir + "/" + singledir + "/" + str(index) + "/"
                    print (str(data_subset_dir))
		    for subparents, subdirnames, subfilenames in os.walk(data_subset_dir):
                        savedir = data_subset_dir+'generated'
                        if not os.path.exists(savedir):
                            os.mkdir(savedir)
		        for subfilename in subfilenames:
                            resize_all(index, data_subset_dir, subfilename, savedir)

    dataarr = np.array(datalist, dtype = np.uint8)
    np.set_printoptions(threshold = np.inf)  

    data['batch_label'.encode('utf-8')]='testing batch 1 of 1'.encode('utf-8')
    data.setdefault('data'.encode('utf-8'), dataarr)    
    data.setdefault('filenames'.encode('utf-8'), filenamelist)
    data.setdefault('labels'.encode('utf-8'), labellist)

    output = open(OUTPUT_BATCH_PATH, 'wb')
    p.dump(data, output)
    output.close()

'''
    x_all = []
    y_all = []
    f_all = []

    d = unpickle(OUTPUT_BATCH_PATH)
    x_all.append(d['data'])
    y_all.append(d['labels'])
    f_all.append(d['filenames'])

    print (np.shape(x_all))

    x = np.concatenate(x_all) / np.float32(255)
    y = np.concatenate(y_all)

    print (x.shape)
    x = np.dstack((x[:, :picsize], x[:, picsize:picsize*2], x[:, picsize*2:picsize*3]))
    x = x.reshape((x.shape[0], length, width, 3))

 
    fo = open(OUTPUT_BATCH_PATH, 'rb')
    dict1 = p.load(fo)
    fo.close()
    print dict1
'''

def unpickle(file):
    fo = open(file, 'rb')
    dict = p.load(fo)
    fo.close()
    return dict


def resize_all(index, cwd, currentpng, savedir):
    outpath = "new_" + currentpng
    currentpath = cwd + currentpng
    im = Image.open(currentpath)
    outname=cwd+outpath
    x_s = length
    y_s = width
    print im
    out = im.resize((x_s,y_s),Image.ANTIALIAS)
    out.save(outname)
    print ("saved")

    outim = Image.open(outname)
    print outim

    list0 = []
    list1 = []
    list2 = []
    listf = []
    for i in range (0, x_s):
        for j in range (0,y_s):
            cl = im.getpixel((i,j))
            list0.append(cl[0])
            list1.append(cl[1])
            list2.append(cl[2])

    listf = list0 + list1 + list2

    os.remove(outname)
    #print "removed"

    datalist.append(listf)
    labellist.append(index)
    filenamelist.append(currentpng)
'''
    print ("data:")
    print (listf)
    print("filename:")
    print(currentpng)
    print("label:")
    print(index)
''' 
 


def one_hot(i):
    a = np.zeros(NUM_CLASS, dtype = int)
    a[i] = 1
    return a

if __name__ == '__main__':
    main(sys.argv)
