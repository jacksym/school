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
        peaks_ind = sig.find_peaks(self.strength, height=400, distance=70, width=1)[0][:4]
        for i in peaks_ind: ticks.append(self.wavelengths[i])
        ax.set_xticks(ticks)
        ax.plot(self.wavelengths, self.strength, linewidth=0.5, color=color,label=label)




fig, ax = plt.subplots()
ax.axhline(0,linewidth=0.5,linestyle='--', color='black')
ax.set_xlim(340,620)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Strength (counts/second)')
ax.set_title('Traces of LEDs on x1 SYBR1')
ticks=[]
ax.grid(True)

blue = Spectrum('50_3.txt')
uv = Spectrum('50_3_UV.txt')

blue.plot(ax, 'blue', 'Blue LED')
uv.plot(ax, 'black', 'UV LED')

ax.legend(loc='upper right')
fig.savefig('../images/stain2.png', quality=100)
plt.show()
