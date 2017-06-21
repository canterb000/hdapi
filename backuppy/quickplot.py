#!/usr/bin/env python
import sys
from matplotlib import pyplot as pl
import os

def main(argv):
    input_array = []
    file_name = argv[1]
    with open(file_name, 'r') as f:
        for data in f.readlines():
            input_array.append(float(data))
            print float(data)

    pl.plot(input_array)
    pl.show()


if __name__ == '__main__':
    main(sys.argv)

