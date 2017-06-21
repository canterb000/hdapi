from scipy import signal  
import numpy as np  
import matplotlib.pyplot as pl  
import matplotlib  
import math


printline = 0
x = []
filtered = []
MEDIAN_WIN = 200

subtracted = []

with open('output_yao.txt', 'r') as f:
    for line in f.readlines(): 
            printline += 1
            v = float(line)
            x.append(v)
            if len(x) > MEDIAN_WIN:
                subset = x[printline - MEDIAN_WIN : printline]
                subset = sorted(subset) 
                newsig = subset[MEDIAN_WIN/2]
                subsig = x[printline - MEDIAN_WIN/2] - newsig
            else:
                newsig = v
                subsig = 0
            filtered.append(newsig)
            subtracted.append(subsig)



axis_x = np.linspace(0,1,num=printline)

pl.subplot(221)  
pl.plot(axis_x,x)


pl.subplot(222)
pl.plot(axis_x,subtracted)





pl.show()


 
