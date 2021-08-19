import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = "./ratedata/"
d1 = pd.read_csv(path+"d1.tsv", sep='\t')
d2 = pd.read_csv(path+"d2.tsv", sep='\t')
d3 = pd.read_csv(path+"d3.tsv", sep='\t')
d4 = pd.read_csv(path+"d4.tsv", sep='\t')[:-120]

fig, ax = plt.subplots()
ax.scatter(d3['frame'], d3['area'], marker='.', label='0.501 V', color='salmon')
ax.scatter(d2['frame'], d2['area'], marker='.', label='0.514 V', color='red')
ax.scatter(d4['frame'], d4['area'], marker='.', label='0.539 V', color='firebrick')
ax.scatter(d1['frame'], d1['area'], marker='.', label='0.547 V', color='darkred')
ax.set_title("Cluster Decay over Time")
ax.set_xlabel("frame number")
ax.set_ylabel("pixel count of cluster")
ax.set_xlim(0)
ax.set_ylim(0)

A=-40000
B=5*10**(-3)
C = 40000
label = r'$A\ \tanh (Bt)+C$'
x = np.linspace(0, 1000, 100)
area = A*(np.tanh(B*x))**2 + C
ax.plot(x, area, linestyle='--', color='blue', label=label)
ax.legend()

plt.savefig("decay.png")
plt.show()
