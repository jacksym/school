import numpy as np
import matplotlib.pyplot as plt
# import math
# import pandas as pd
import os
# import time

path = "./ogle/"
events = sorted(os.listdir('./ogle/'))[0:1000:300]
dat = '/phot.dat'


for event in events:
    df = np.loadtxt(path+event+dat).transpose()
    df[0] = df[0] - 2450000

    i_mean = np.mean(df[1])

    m_max, m_min = np.argmax(df[1]), np.argmin(df[1])
    i_max, i_min = df[1][m_max] + df[2][m_max], df[1][m_min] - df[2][m_max]
    sep = 0.2*(i_max-i_min)

    plt.figure(figsize=(10, 7))
    plt.axhline(i_mean, color='gray', linewidth='0.5')
    plt.title(r'OGLE-2019 $\rightarrow$ '+event)
    plt.xlabel("HJD - 2450000")
    plt.ylabel(r'$I$ magnitude')
    # plt.scatter(df[0], df[1], marker='.')
    plt.errorbar(df[0], df[1], df[2], fmt='none', color='black', linewidth=0.5)
    plt.plot(df[0], df[1], color='gray', linewidth=0.3)
    plt
    plt.ylim(i_max+sep, i_min-sep)
    plt.show()
    plt.close()
