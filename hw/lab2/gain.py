import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_trendline(x, y, axes):
    m, b = np.polyfit(x, y, 1)
    xval = np.linspace(np.min(x), np.max(x), 10)
    axes.plot(xval, m*xval+b, color='red', linestyle='--')

gdf = pd.read_csv('va.tsv', sep='\t')
vpp = gdf['Vpp']
vin = gdf['Vin']
vout = gdf['Vout']*1000
gain = vout/gdf['Vin']

fig, ax = plt.subplots(2,1)
plt.subplots_adjust(hspace=0.5)

ax[0].scatter(vpp, vin, label='Input Voltage')
ax[0].scatter(vpp, vout/1000, label=r'Output Voltage $\times \frac{1}{1000}$')
ax[0].set_title('In and Out Voltages of the Amplifier')
ax[0].set_xlabel(r'$V_\mathrm{pp}$ of AC Function Generator')
ax[0].set_ylabel('Voltage (V)')
ax[0].legend()

ax[1].grid(True)
ax[1].scatter(vpp, gain, marker='o', color='red')
ax[1].set_title('Gain')
ax[1].set_xlabel(r'$V_\mathrm{pp}$ of AC Function Generator')
ax[1].set_ylabel(r'$V_\mathrm{out}/V_\mathrm{in}$')
get_trendline(vpp, gain, ax[1])

plt.savefig('./images/va.png')
plt.show()
