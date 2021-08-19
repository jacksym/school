import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

class Spectrum:
    def __init__(self, filename):
        with open(filename) as text:
            lines = text.readlines()
            int_time = float(lines[6].split()[-1])
            wv = []
            strength = []
            for line in lines[14:]:
                line = line.split()
                wv.append(float(line[0]))
                strength.append(float(line[1]))

        self.wavelengths = np.array(wv)
        self.strength = np.array(strength)/int_time

    def plot(self, ax, color, label):
        ax.plot(self.wavelengths, self.strength, linewidth=0.5, color=color,label=label)




fig, ax = plt.subplots()
ax.axhline(0,linewidth=0.5,linestyle='--', color='black')
ax.set_xlim(400,670)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Strength (counts/second)')
ax.grid(True)

full_conc = Spectrum('full_conc.txt')
conc_80_1 = Spectrum('80_1.txt')
conc_80_2 = Spectrum('80_2.txt')
conc_50_1 = Spectrum('50_1.txt')
conc_50_2 = Spectrum('50_2.txt')

full_conc.plot(ax,'black', '1')
conc_80_1.plot(ax,'black','1') 
conc_80_2.plot(ax,'black','1') 
conc_50_1.plot(ax,'black','1') 
conc_50_2.plot(ax,'black','1') 

fig.savefig('../images/conc2.png', quality=100)
plt.show()
