import matplotlib.pyplot as plt
import numpy as np

data = []

with open('loss.txt', 'rb') as f:
#    for x in f.read().split('\n'):
#        if x.isdigit():
#            data.append(float(x))
    data = f.readlines()
    
    
#data = np.array(data)
#data = data.transpose()
#plt.figure
#plt.plot(data)
