#Bright Ogle
#Jack Symonds 
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import data_proc
import os
import time

path = "./ogle/"

events = os.listdir(path)

event_list = []
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
with open("err_cor_f.dat", 'w') as file:
    for event in events:
        time_b = time.time()
        try:
            event_dat = data_proc.OGLE_data(path+event)
            event_dat.plot_f(ax, event)
            mean = event_dat.mean_const
            model = event_dat.gauss_model_ec(event_dat.diffs, ax2)
            cp = event_dat.cp
            np.savetxt(file, np.array([mean, cp]))
            np.savetxt('err_cor_f.data', np.array([mean, cp]), newline='\n')
            print(np.array([mean, cp]))
        except:
            print('error')

# event_list = sorted(event_list, key=lambda x: x[1])
# dims = event_list[0:10]
# brights = event_list[-11:-1]


