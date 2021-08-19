import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sig

class temptime:
    def __init__(self, filename):
        self.df = pd.read_csv('./pid/'+filename)

    def plot(self, ax, color,P,I, tstab,tset):
        times = self.df['Time (s)']
        temps = self.df['Temperature (K)']

        ax.axvline(tstab, linewidth=0.8, linestyle='--', color=color)

        #ax.fill_between(times, temps, avgT, color='lightgrey')

        tstabi = np.argmin(times-tstab)
        ntemps = temps[tstabi:]
        rms = np.sqrt((1/(ntemps.size-1))*np.sum((ntemps-np.mean(ntemps))**2))
        label = r'I = %.2f' % I
        ax.plot(times, temps, marker='', color=color,label=label,linewidth=0.8)
        row = [r'$P=%.2f$' % P, r'$I=%.2f$'%I, r'$t_{stab}\approx %.2f$ s'%round(tstab,2), r'$\sigma _T=%.4f$' % rms]
        cellText.append(row)


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.16)
ax.grid(True)
ax.set_xlabel('Time (s)')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_ylabel('Temperature (K)')
cellText=[]

#pi = temptime('vp2.csv')
#pi.plot(ax, 'black',1.6, 4.4, 10)






pi = temptime('p1-6.csv')
pi.plot(ax, 'black',1.6, 4.4, 30,10)

vi = temptime('i2-2.csv')
vi.plot(ax, 'blue',1.6,2.2,70,10)

vi = temptime('i8-8.csv')
vi.plot(ax, 'orange',1.6,8.8,25,10)


plt.legend()
ax.table(cellText=cellText, cellLoc='left')
fig.savefig('images/itrace.png')
plt.show()
