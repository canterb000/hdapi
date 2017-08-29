#!/usr/bin/env python

import os
import tensorflow as tf 
import sys
import array
import numpy as np
#from PIL import image
#import Image
from PIL import Image 
import cPickle as p
#import chardet


def main(argv):
    cwd = os.getcwd()
    print (cwd)

    currentpath = cwd +"/original.png"
    im = Image.open(currentpath)
    #width=im.size[0]
    #height=im.size[1]
    x_s = 32
    y_s = 32
    out = im.resize((x_s,y_s),Image.ANTIALIAS)
    out.save(cwd +"/new.png")
    new_im = Image.open(cwd+"/new.png")
    new_im.show()

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

    listf.append(list0)
    listf.append(list1)
    listf.append(list2)


    np.set_printoptions(threshold=np.inf)  
    arr2 = np.array(listf, dtype = np.uint8)
    print (arr2)

def one_hot(i):
    a = np.zeros(NUM_CLASS, dtype = int)
    a[i] = 1
    return a

if __name__ == '__main__':
    main(sys.argv)
