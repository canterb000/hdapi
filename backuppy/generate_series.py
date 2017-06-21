#!/usr/bin/env python

import os
import sys
from matplotlib import pyplot as pl
import numpy as np
from scipy import signal 

SAMPLE_RATIO = 5


def main(argv):


    current_dir = os.getcwd();
    print "current_dir: " + current_dir
    for parent, dirnames, filenames in os.walk(current_dir):
        for singlefile in filenames:
            file_name, file_extension = os.path.splitext(singlefile)
            if file_extension == ".dat":
                print "dat file found: " + singlefile
                ecgparse(singlefile)
 
def ecgparse(singlefile):
    file_name, file_extension = os.path.splitext(singlefile)

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

    data_buffer = np.fromfile(singlefile,dtype=np.uint8)
    data_len = np.frombuffer(data_buffer,dtype=np.uint32,count = 1,offset=FILE_LEN_POSITION)
    data_len = data_len[0]
    data = np.int32(np.frombuffer(data_buffer,dtype=DATA_TYPE,count=data_len/DATA_SIZE_BYTE,offset=FILE_START_POSITION))
    data_scaled = 1000 * (data - (2**BIT_DEPTH / 2 )) * REF_VOLTAGE / (2**BIT_DEPTH)
    data_array = data_scaled.reshape((-1,LEADS_NUM))

    start_point = PLOT_POINT_START
    if PLOT_POINT_STOP is None:
        stop_point = data_len/DATA_SIZE_BYTE/LEADS_NUM/FREQUENCE * 1000
    else:
        stop_point = PLOT_POINT_STOP

    t = np.linspace(0,(stop_point - start_point)/1024. *1000, stop_point - start_point)

    output_array = data_array[start_point:stop_point,1]

    resampled_array = signal.resample(output_array, len(output_array)/SAMPLE_RATIO)

    print  len(resampled_array)
#    pl.plot(t, output_array, linewidth=3)
#    pl.plot(t[::SAMPLE_RATIO], resampled_array, 'ko')

#    for i in range(1,LEADS_NUM):
#        if i <= LEADS_NUM/2 :
#            pl.subplot(LEADS_NUM/2,2, 2*i-1 )
#        else:
#            pl.subplot(LEADS_NUM/2,2, 2*(i-LEADS_NUM/2) )

#        pl.plot(t,data_array[start_point:stop_point,i])
#        pl.grid(True)
#    pl.show()

#    offset = 0;
#    if offset > (len(output_array)-end_offset):
#            offset = random.randint(0, n_input+1)


#    next = input("Press Enter:")
#    print next


    generate_datasets(resampled_array, file_name) 
    

#    print "export data to:" + outputfile
#    for data in output_array:
#        output.write(str(data)+"\n")


def generate_datasets(input_array, input_name):

    SINGLE_FILE_LEN = 500

    end_point = len(input_array)
    current = 0
    while current < (end_point - SINGLE_FILE_LEN):

        outputfile  = "output_" + input_name + str(current) + ".txt"
        output = open(outputfile, 'w')

        print "export data to:" + outputfile + "current:" + str(current)
 
        for data in input_array[current:current+SINGLE_FILE_LEN]:
            output.write(str(data)+"\n")
#        next = input("Press Enter:")
#        print next


        output.close()
        current += 10


if __name__ == '__main__':
    main(sys.argv)

