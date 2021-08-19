import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

boltz = 1.380649*10**(-23)
room_temp = 293
elem = 1.60217662*10**(-19)

class VT:
    def __init__(self, filename):
        path = './iv/'
        self.df = pd.read_csv(path+filename, sep='\t').sort_values(by='T')
        self.t = self.df['T']
        self.v = self.df['V'] - 0.001*self.df['Va']
    
    def plot(self, ax, label, color):
        label = r'$I\approx %3d$'%label
        ax.errorbar(self.t, self.v, xerr=0.005, fmt='none', label='_nolabel_')

        lx = self.t.min()
        mx = self.t.max()
        m, b = np.polyfit(self.t, self.v, 1)
        x = np.linspace(lx*0.95, mx, 10)
        trend = r'$\qquad V = %.3f\ T %+.3f$' % (m, b)
        ax.scatter(self.t, self.v, label=label, color=color)
        ax.plot(x, m*x + b, linestyle='--',color=color)
        Eg = b*elem/elem
        row = [label,trend,r'$E_g=%.3f$ eV'%Eg]
        global cellText
        cellText.append(row)
        plt.legend()

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.15)
ax.grid(True)
ax.set_xlabel('Temperature (K)')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_ylabel('Voltage (V)')
ax.axhline(0, color='black', linewidth=0.75)
cellText=[]

vt50 = VT('vt50.tsv')
vt200 = VT('vt200.tsv')

vt50.plot(ax,50,'grey')
vt200.plot(ax,200,'black')

plt.table(cellText=cellText, cellLoc='left')
fig.savefig('images/vt.png')
plt.show()
