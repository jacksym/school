import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

bohrmag = 9.274*10**(-24)
planck = 6.626*10**(-34)

def magForI(currents):
    B = 249+8.2999*currents+3.9277*10**(-3)*currents**2 - 5.5726*10**(-6)*currents**3
    B /=10000
    return B

class Zeem:
    def __init__(self, filename):
        df = pd.read_csv(filename, sep='\t')
        self.b = magForI(df['I'])
        self.b = sorted(np.append(self.b, 0))
        self.df = df['df']
        self.df = sorted(np.append(self.df, 0))

    def plot(self, axes, label, color):
        axes.scatter(self.b, self.df, marker='.', color=color, label='_nolabel_')

        fit, cov = np.polyfit(self.b, self.df, 1, cov=True)
        u = np.sqrt(np.diag(cov))
        print(u)
        lx = np.min(self.b)
        mx = np.max(self.b)

        m, b = fit
        x = np.linspace(0, mx, 10)
        label = str(label) + r'$\qquad$'
        label += r'$\Delta f={}\ B{}{}\qquad U={}$'.format(round(m, 4), '+' if b>0 else '', round(b, 4), u)

        g = planck*3*10**8*m/(bohrmag*1*2*10.06*10**(-3))
        label += r'$\qquad g = {}$'.format(round(g, 4))

        axes.plot(x, m*x+b, color=color, linestyle='--', label=label)
    
        


fig, ax = plt.subplots()
ax.grid(True)
ax.set_ylim(0, 0.75)
ax.set_xlim(0, 0.75)
ax.set_yticks([0,0.25,0.5])
#ax.set_yticklabels(['0', r'$\frac{\Delta f_{fsr}}{4}$', r'$\frac{\Delta f_{fsr}}{2}$'])
ax.set_yticklabels(['0', r'$\frac{1}{4}$', r'$\frac{1}{2}$'])
ax.set_xlabel('Magnetic Field (T)')
ax.set_ylabel('Fraction of Free Spectral Range of Shift')

w585 = Zeem('585.tsv')
w585.plot(ax, 585, 'black')
plt.legend()
plt.savefig('./images/zeem1.png')

w607 = Zeem('607.tsv')
w607.plot(ax, 607, 'gray')

w616 = Zeem('616.tsv')
w616.plot(ax, 616, 'darkgray')

w653 = Zeem('653.tsv')
w653.plot(ax, 653, 'silver')

ax.legend(loc='upper right')
plt.savefig('./images/zeem.png')
plt.show()
