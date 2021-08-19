import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = "./ratedata/"
g1 = pd.read_csv(path+"g1.tsv", sep='\t')
g2 = pd.read_csv(path+"g2.tsv", sep='\t')
g3 = pd.read_csv(path+"g3.tsv", sep='\t')
g4 = pd.read_csv(path+"g4.tsv", sep='\t')

fig, ax = plt.subplots()
ax.scatter(g3['frame'], g3['area'], marker='.', label='0.400 V', color='darkgreen')
ax.scatter(g4['frame'], g4['area'], marker='.', label='0.417 V', color='green')
ax.scatter(g1['frame'], g1['area'], marker='.', label='0.440 V', color='limegreen')
ax.scatter(g2['frame'], g2['area'], marker='.', label='0.446 V', color='gray')
ax.set_title("Cluster Growth over Time")
ax.set_xlabel("frame number")
ax.set_ylabel("pixel count of cluster")
ax.set_xlim(0)
ax.set_ylim(0)

A=40000
B=2*10**(-3)
C = 1000
label = r'$A\tanh (Bt)+C$'
x = np.linspace(0, 1000, 100)
area = A*(np.tanh(B*x))**2 + C
ax.plot(x, area, linestyle='--', color='blue', label=label)
ax.legend()
plt.savefig("growth.png")
plt.show()
