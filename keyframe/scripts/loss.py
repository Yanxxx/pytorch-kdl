import matplotlib.pyplot as plt
import numpy as np


data = []

path = 'loss/20210707-215045/'
loss = []

with open(path + 'loss.txt', 'rb') as f:
#    for x in f.read().split('\n'):
#        if x.isdigit():
#            data.append(float(x))
    data = f.readlines()
    for i in data:
        loss.append(float(i))
    
    
#data = np.array(data)
#data = data.transpose()
plt.figure
plt.plot(loss)
