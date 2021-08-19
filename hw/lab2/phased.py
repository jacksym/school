import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("phased.tsv", sep='\t')
#df = df.append(pd.DataFrame([[0,0,0]], columns=['f', 'v1', 'v2']))
df = df.sort_values('f')
thickness = 1.62*10**(-3)
f = df['f']
E1 = df['v1']/thickness
E2 = df['v2']/thickness

fig, ax = plt.subplots()

ax.grid(True)
ax.plot(f, E1, linestyle='--', marker='o', color='g')
ax.plot(f, E2, linestyle='--', marker='o', color='r')
ax.fill_between(f, E2, E1, color='tan')
ax.set_title('Empirical Phase Diagram of Nickel Powder')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel(r'Electric Field (kVm$^{-1}$)')
ax.set_xlim(0, np.max(f))
ax.set_ylim(0, 350)
ax.text(50, 320, 'Gas (clusters can not form)', fontsize=14)
ax.text(50, 250, 'Coursening (clusters and gas coexist)', fontsize=14)
ax.text(50, 100, 'No Motion', fontsize=14)

rho = 8908
a = 75*10**(-6)
g = 9.8
ep0 = 8.854*10**(-12)
k =1.36
tE1 = np.sqrt((rho*a*g)/(3*ep0*k))/1000
print(tE1)
ax.plot(f, (0*f)+tE1, linestyle='--', color='blue')

plt.savefig("images/phased.png")
plt.show()
