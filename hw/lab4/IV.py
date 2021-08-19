import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

boltz = 1.380649*10**(-23)
room_temp = 293
elem = 1.60217662*10**(-19)

class IV:
    def __init__(self, filename):
        path = './iv/'
        self.df = pd.read_csv(path+filename, sep='\t').sort_values(by='V')
        self.v = self.df['V'] - 0.001*self.df['Va']
        self.i = self.df['I']
    
    def plot(self, ax, label):
        label = '{} K'.format(label)
        ax.scatter(self.v, self.i, label=label)
        ax.errorbar(self.v, self.i, xerr=0.005, fmt='none', label='_nolabel_')

        #lx = self.v.min()
        #mx = self.v.max()
        #m, b = np.polyfit(self.v, self.i, 1)
        #x = np.linspace(lx*0.95, mx, 10)
        #label = r'$I=%1.2f \ V%1.2f\qquad V_\mathrm{int}=%1.3f$' % (m, b, vint)
        #ax.plot(x, m*x + b, linestyle='--')
        plt.legend()

fig, ax = plt.subplots()
ax.grid(True)
ax.set_title(r'$I-V$ Curve')
ax.set_xlabel('Voltage (V)')
ax.set_ylabel(r'Current ($\mu$A)')
ax.axhline(0, color='black', linewidth=0.75)
 
#iv10 = IV('IV10.tsv')
#iv10.plot(ax, 10)
iv50 = IV('IV50.tsv')
iv100 = IV('IV100.tsv')
iv150 = IV('IV150.tsv')
iv200 = IV('IV200.tsv')
iv250 = IV('IV250.tsv')
ivrt = IV('IVrt.tsv')

iv250.plot(ax, 250)
iv50.plot(ax, 50)
iv100.plot(ax, 100)
iv150.plot(ax, 150)
iv200.plot(ax, 200)
ivrt.plot(ax, 294.5)

fig.savefig('./images/iv.png')
plt.show()
