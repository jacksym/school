import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fig, ax = plt.subplots(2,2)
plt.subplots_adjust(wspace=0.5, hspace=0.5)

def get_trendline(x, y, axes):
    m, b = np.polyfit(x, y, 1)
    xval = np.linspace(np.min(x), np.max(x), 10)
    axes.plot(xval, m*xval+b, color='red', linestyle='--')

def plots(filename, column, title):
    gdf = pd.read_csv(filename, sep='\t').sort_values('f')
    f = gdf['f']
    vin = gdf['Vin']
    vout = gdf['Vout']*1000
    gain = vout/gdf['Vin']

    ax[0,column].grid(True)
    ax[0,column].scatter(f, vin, label='Input Voltage', marker='.')
    ax[0,column].scatter(f, vout/1000, label=r'Output Voltage $\times \frac{1}{1000}$', marker='.')
    ax[0,column].set_title(title)
    ax[0,column].set_xlabel('Frequency of AC Function Generator')
    ax[0,column].set_ylabel('Voltage (V)')
    ax[0,column].legend()

    ax[1,column].grid(True)
    ax[1,column].scatter(f, gain, marker='o', color='red')
    ax[1,column].set_title('Gain')
    ax[1,column].set_xlabel(r'$V_\mathrm{pp}$ of AC Function Generator')
    ax[1,column].set_ylabel(r'$V_\mathrm{out}/V_\mathrm{in}$')
    #get_trendline(f, gain, ax[1,column])

plots('vf.tsv', 0, r'$V_\mathrm{in}$ & $V_\mathrm{out}$ for $V_\mathrm{pp}$ 2.0 V')
plots('vf2.tsv', 1, r'$V_\mathrm{in}$ & $V_\mathrm{out}$ for $V_\mathrm{pp}$ 7.0 V')

plt.savefig('./images/vf.png')
plt.show()
