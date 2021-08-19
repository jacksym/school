import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

boltz = 1.380649*10**(-23)
room_temp = 293
elem = 1.60217662*10**(-19)

class IV:
    def __init__(self, filename):
        self.df = pd.read_csv(filename, sep='\t').sort_values(by='V')
        self.v = self.df['V']
        self.i = self.df['I']
    
    def ivPlot(self, ax, color, points):
        ax.scatter(self.v, self.i, color=color, label='_nolabel_')
        #self.ax.errorbar(self.i, self.v, yerr=self.df['Ve'], fmt='none')

        lx = self.v[-(points+2):].min()
        mx = self.v.max()
        m, b = np.polyfit(self.v[-points:], self.i[-points:], 1)
        x = np.linspace(lx*0.95, mx, 10)
        vint = -b/m
        print(elem*vint)
        label = r'$I=%1.2f \ V%1.2f\qquad V_\mathrm{int}=%1.3f$' % (m, b, vint)
        ax.plot(x, m*x + b, linestyle='--', color=color, label=label)

    def logIvPlot(self, ax, color, points):
        ax.grid(True)
        x = self.v[1:points]
        y = np.log(self.i[1:points])
        ax.scatter(x, y, color=color, label='_no_label_')

        lx = np.min(x)
        mx = np.max(x)
        m, b = np.polyfit(x, y, 1)
        n = elem/(m*boltz*room_temp)
        xval = np.linspace(lx, mx, 10)
        label = r'$\ln (I)={}\ V{}\qquad n = {}$'.format(round(m,3), round(b,3), round(n, 3))
        ax.plot(xval, m*xval + b, linestyle='--', color=color, label=label)

blue = IV("bIVR.tsv")    
orange = IV("oIVR.tsv")
red = IV("rIVR.tsv")
ir = IV("irIVR.tsv")

fig, ax = plt.subplots()
ax.grid(True)
ax.set_title(r'$I$-$V$ Curve for Non-White LEDs')
ax.set_xlabel('Voltage (V)')
ax.set_ylabel('Current (mA)')
ax.set_ylim(-1, 15)
ax.axhline(0, color='black', linewidth=0.75)

ir.ivPlot(ax, "black", 5)
red.ivPlot(ax, "red", 5)
orange.ivPlot(ax, "orange", 5)
blue.ivPlot(ax, "blue", 4)
plt.legend()
plt.savefig('./images/linear_IV.png')

fig2, ax2 = plt.subplots()
ax2.set_xlabel('Voltage (V)')
ax2.set_ylabel(r'$\ln(I)$')
ax2.set_ylim(-5,4)
ir.logIvPlot(ax2, "black", 6)
red.logIvPlot(ax2, "red", 6)
orange.logIvPlot(ax2, "orange", 7)
blue.logIvPlot(ax2, "blue", 7)
plt.legend(loc='upper right')
plt.savefig('./images/log_IV.png')
plt.show()
