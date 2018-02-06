import wfdb
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from IPython.display import display
from findtools.find_files import (find_files, Match)
import ntpath
import random

endofsample = 100000
#filedir = "e0103"
#onlinedir="svdb/"
databasedir = "ecgtraindatabase/"
databasetestdir = "ecgtestdatabase/"
originfiledir="originalfiles/"
halfoffset = 25
halfoffset1 = 50
halfoffset2 = 90

totalsize = 0
recordsize = 0


#dblist = wfdb.getdblist()
#print(dblist)
#os._exit()

Ncount = 0
Vcount = 0
Scount = 0
countsum = [[0 for x in range(4)] for y in range(2)] 
countsum[1][3] = 0

equalmaxcount = 4000

txt_files_pattern = Match(filetype = 'f', name = '*.dat')
found_files = find_files(path=originfiledir, match=txt_files_pattern)
for found_file in found_files:
    head, tail = ntpath.split(found_file)
    recordname = tail.split('.')[0]
    print head
    print tail
    print recordname
    recordsize = recordsize + 1
    readdir = head + '/' + recordname

#    record = wfdb.rdsamp(readdir, sampto = endofsample)
    record = wfdb.rdsamp(readdir)
    annotation = wfdb.rdann(readdir, 'atr', sampfrom = 0, sampto = endofsample)
#    display(annotation.sample)
#   print("list of samples:")
    num = len(annotation.sample)
    i = 0
    while i < num:

        label = 3
#        print ("iter:  %d" % i)
#        print ("location: %d" % annotation.sample[i])
#        print ("label: %s " % annotation.symbol[i])
        if annotation.symbol[i] == "+":
 #           print "skiping --- + symbol"
            i = i + 1
            continue
        if annotation.symbol[i] == "N":
            label = 0
            if Ncount > equalmaxcount:
                i = i + 1
                continue
            else:
                Ncount = Ncount + 1
        if annotation.symbol[i] == "V":
            label = 1
            if Vcount > equalmaxcount:
                i = i + 1
                continue
            else: 
                Vcount = Vcount + 1
        if annotation.symbol[i] == "S":
            label = 2
            if Scount > equalmaxcount:
                i = i + 1
                continue
            else:
                Scount = Scount + 1
        if Ncount+Scount+Vcount > (3 * equalmaxcount):
            break
        
        #print ('{}{}{}{}{}'.format(Ncount, ",", Vcount, ",", Scount))

        loc = annotation.sample[i]
        start = max(loc - halfoffset1, 0)
        end = loc + halfoffset2
        record = wfdb.rdsamp(readdir,  sampfrom = start, sampto = end)
        #print(record.p_signals)
        if not os.path.isdir(databasedir):
            print(os.mkdir(databasedir))
        if not os.path.isdir(databasetestdir):
            print(os.mkdir(databasetestdir))
        if not os.path.isdir('{}{}'.format(databasedir, label)):
            print (os.mkdir('{}{}'.format(databasedir, label)))
        if not os.path.isdir('{}{}'.format(databasetestdir, label)):
            print (os.mkdir('{}{}'.format(databasetestdir, label)))

        plt.plot(record.p_signals[:,0])
        plt.axis('off')
        '''	
	if annotation.symbol[i] == 's':
	    print "lowercase s"
        elif annotation.symbol[i] == 'S':
            print "uppercase S"
        '''
        
#        print('{}{}{}'.format(annotation.symbol[i], "#", label))

        randomchoice = random.uniform(0,10)
        if randomchoice < 5: 
            countsum[1][label] = countsum[1][label] + 1
            savedfigurepath = '{}{}{}{}{}{}{}'.format(databasetestdir, label, "/", recordname, "_", annotation.sample[i], ".png")
        else: 
            countsum[0][label] = countsum[0][label] + 1
            savedfigurepath = '{}{}{}{}{}{}{}'.format(databasedir, label, "/", recordname, "_", annotation.sample[i], ".png")
#        print("saving as %s" % savedfigurepath)
        plt.savefig(savedfigurepath)
        plt.clf() 
        totalsize = totalsize + 1
        i = i + 1
        if totalsize % 100 == 0:
            print (totalsize)
#print ("size of the database:%d, N:%d, V:%d, S:%d" % totalsize Ncount  Vcount Scount)
print ("size of the database:%d, N:%d, V:%d, S:%d" % (totalsize ,  Ncount ,  Vcount,  Scount))

print ("size of the testdatabase:%d, N:%d, V:%d, S:%d, other:%d" % (sum(countsum[1][0:3]),  countsum[1][0],  countsum[1][1],  countsum[1][2], countsum[1][3]))
print ("size of the traindatabase:%d, N:%d, V:%d, S:%d, other:%d" % (sum(countsum[0][0:3]), countsum[0][0],  countsum[0][1],  countsum[0][2], countsum[0][3]))


outfile = open("outfile.txt","w")
outfile.write('{}{}{}{}{}{}{}{}'.format("testsize:", (sum(countsum[1][0:3])), ",  N:", countsum[1][0], ",  V:", countsum[1][1], ", S:", countsum[1][2]))
outfile.write('\n')
outfile.write('{}{}{}{}{}{}{}{}'.format("trainsize:", (sum(countsum[0][0:3])), ",  N:", countsum[0][0], ",  V:", countsum[0][1], ", S:", countsum[0][2]))

outfile.close()

'''
print ("symbol-------------------------------")
display(annotation.symbol)
print ("subtype-------------------------------")
display(annotation.subtype)
print ("associated channel-------------------------------")
display(annotation.chan)
print ("annotation number-------------------------------")
display(annotation.num)

wfdb.plotrec(record,  title='Online Database', timeunits = 'seconds', ecggrids = 'all')
#wfdb.plotrec(record,  title='Online Database', annotation = annotation, timeunits = 'seconds', ecggrids = 'all')
'''
