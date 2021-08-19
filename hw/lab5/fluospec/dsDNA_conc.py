import numpy as np
import matplotlib.pyplot as plt

def get_peak(filename):
    with open(filename) as text:
        lines = text.readlines()
        int_time = float(lines[6].split()[-1])
        wv = []
        strength = []
        for line in lines[14:]:
            line = line.split()
            wv.append(float(line[0]))
            strength.append(float(line[1]))

    strength = np.array(strength)/int_time
    peak = np.max(strength)
    strengths.append(peak)
    peak_wav.append(wv[np.argmax(strength)])

strengths = []
peak_wav = []

get_peak('full_conc.txt')
get_peak('80_1.txt')
get_peak('80_2.txt')
get_peak('50_1.txt')
get_peak('50_2.txt')
get_peak('50_3.txt')

conc = []; c = 1
for i in [100,80,80,50,50,50]:
    c *= (i/100)
    conc.append(c)
conc = np.array(conc)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.15)
ax.grid(True)
ax.scatter(conc, strengths)
ax.set_xlabel(r'Concentration (relative to 60 $\mu$l in 2 ml of SYBR1/PBS)')
ax.set_ylabel('Strength of Peak Wavelength in Spectrum')

fig.savefig('../images/dsDNA_conc.png', quality=100)
plt.show()
