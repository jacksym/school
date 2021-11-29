import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
# import pandas as pd
# import os


path = "./ogle/"

file = "blg-0003/phot.dat"

df = np.loadtxt(path+file).transpose()
df[0] = df[0] - 2450000

i_mean = np.mean(df[1])


m_max, m_min = np.argmax(df[1]), np.argmin(df[1])
i_max, i_min = df[1][m_max] + df[2][m_max], df[1][m_min] - df[2][m_max]
sep = 0.2*(i_max-i_min)
# g_max, g_min = math.ceil((i_max+sep)*4)/4, math.floor((i_min-sep)*4)/4


fig, ax = plt.subplots()

ax.axhline(i_mean, color='gray', linewidth='0.5')
ax.set_title(r'OGLE-2019 $\rightarrow$ '+file)
ax.set_ylim(i_max+sep, i_min-sep)
ax.set_xlabel("HJD - 2450000")
ax.set_ylabel(r'$I$ magnitude')
ax.scatter(df[0], df[1], marker='.')
# ax.errorbar(df[0], df[1], df[2], fmt='none', color='black', linewidth=0.5)

plt.show()
