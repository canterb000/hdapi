from matplotlib import pyplot as pl
import numpy as np

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

ecg_file = 'yao.dat'
data_buffer = np.fromfile(ecg_file,dtype=np.uint8)

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
for i in range(0,LEADS_NUM-1):
    if i < LEADS_NUM/2 :
        pl.subplot(LEADS_NUM/2,2, 2*i+1 )
    else:
        pl.subplot(LEADS_NUM/2,2, 2*(i+1-LEADS_NUM/2) )
    
    pl.plot(t,data_array[start_point:stop_point,i])
    pl.grid(True)
pl.show()

