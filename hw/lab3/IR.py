import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fig, ax = plt.subplots()

class IR:
    def __init__(self, filename, intTime):
        self.df = pd.read_csv(filename, sep='\t').dropna().sort_values(by='I')
        self.i = self.df['I']
        self.r = self.df['R']/(1000*intTime)
    
    def irPlot(self, color):
        ax.grid(True)
        ax.set_xlabel('Current (mA)')
        ax.set_ylabel(r'Absolute Irradiance (mW)')
        #ax.errorbar(self.i, self.r, yerr=self.df['Re'], fmt='none')

        m, b = np.polyfit(self.i, self.r, 1)
        x = np.linspace(0,10,5)
        label = r'$R={}\ I{}{}$'.format(round(m,3),'+' if b > 0 else '', round(b,3))
        ax.plot(x, m*x+b, linestyle='--', color=color, label=label)

        ax.scatter(self.i, self.r, color=color, marker='o', label='_nolabel_')

blue = IR("bIVR.tsv", 1)
orange = IR("oIVR.tsv", 1)
red = IR("rIVR.tsv", 1)
ir = IR("irIVR.tsv", 1)

ax.set_title('Relative Radiative Output of the Four Diodes for Current')
blue.irPlot("blue")
ir.irPlot("black")
red.irPlot("red")
orange.irPlot("orange")
plt.legend()
plt.show()
#plt.savefig('./images/ri.png')


