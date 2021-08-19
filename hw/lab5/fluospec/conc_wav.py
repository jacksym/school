import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

molar_conc = 75.23 #micromolar/m^3

class CvES:
    def __init__(self, filename):
        df = pd.read_csv(filename, sep='\t')
        self.conc = df['conc']*molar_conc/(df['div']+df['conc'])
        self.em = df['em']
        self.strength = df['strength']/(0.001*df['itime'])

    def plot(self, ax):
        ax.set_yscale("linear")
        ax.set_xscale("log")
        ax.scatter(self.conc, self.em)


fig, ax = plt.subplots()
fig.subplots_adjust(left=0.15, bottom=0.15)
ax.grid(True)
ax.set_title('Fluorescein Concentration Peak Wavelength')
ax.set_xlabel(r'Molar Concentration of Sample $\left( \frac{\mu m}{m^3} \right)$')
ax.set_ylabel('Peak Fluorescence Wavelength (nm)')

conc_data = CvES('conc.tsv')

conc_data.plot(ax)

fig.savefig('../images/conc_wav.png')
plt.show()
